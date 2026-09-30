//! Started calls: the runtime side of native call groups (`sky_ir::parallel`).
//!
//! Generated code for a group of independent pure calls (`module.rs`,
//! "Threads") opens a *group*, *spawns* each call that may run elsewhere,
//! *joins* before anything reads a result (and at the end of the group, and on
//! every failure inside it), loads the results, and ends the group. This file
//! is what those runtime calls do.
//!
//! # Where a started call runs
//!
//! A call is started through a *thunk* generated for its callee: a function
//! `(depth, stack limit, call site, args, results) -> status` that unpacks the
//! arguments from memory, makes the ordinary native call and stores the
//! results. Only callees whose parameters and results are numbers, Bools and
//! small arrays start (no handle to a runtime-owned value crosses threads: a
//! handle is an index into the table of one thread's [`State`]).
//!
//! * On a **worker** of the pool the call runs with a runtime state of its own
//!   (its own handle table, pending error and kernel cache), so it shares
//!   nothing with the thread that started it but the program's plain data
//!   (`Arc<Program>`). Its output goes nowhere: a callee with `io` is never
//!   started. The pool is `par::call_pool_for`'s, and a worker never waits
//!   inside rayon (it starts no calls of its own and runs its kernels on
//!   itself), because a rayon worker that waits runs other jobs meanwhile, and
//!   another call's runtime state must never be installed inside a call's.
//! * **In place** (on the calling thread, at once) when the caller has used
//!   more than half of the stack budget (so that a run that would stop with
//!   "the program nests too deeply" stops at the same call), when the caller
//!   already is a worker (workers never wait for other workers, so the pool
//!   cannot deadlock), or when no pool could be started. The call is run by
//!   the same thunk and its outcome is recorded like a worker's.
//!
//! # What a call leaves, and which one wins
//!
//! A started call ends in an *outcome*: its results, its error, or a
//! deoptimization. [`sky_rt_join`] waits for every call started so far,
//! and if any did not succeed it installs the outcome of the **earliest started
//! one** as the run's pending error (replacing whatever is pending: an
//! operation that failed in place after the call started comes later in
//! program order, so the call's failure is the one the sequential run meets
//! first). Calls are started in program order, so "earliest started" is
//! "earliest in program order". A call that stopped because the *worker's*
//! stack budget (half of the run's) ran out is run again on the calling thread
//! at the join, which has the run's stack: then its outcome is the sequential
//! run's.
//!
//! # Unsafe
//!
//! The one `unsafe` operation is calling a thunk by address; see
//! [`call_thunk`].

use std::sync::{Arc, Condvar, Mutex, PoisonError};

use sky_interp::arith::{stack_address, STACK_BUDGET};

use super::*;

/// The signature of a thunk (see the module docs).
type Thunk = unsafe extern "C" fn(i32, i64, i32, *const u64, *mut u64) -> i32;

/// How a started call ended.
pub(super) enum Outcome {
    /// It returned; its results are in [`Done::res`].
    Ok,
    /// It failed with this error.
    Err(RuntimeError),
    /// It found a value that was not what its type says (see the runtime
    /// module docs): the run deoptimizes.
    Deopt,
    /// A worker's stack budget ran out: run it again on the calling thread.
    Retry,
}

/// A finished call: its outcome and its results, as 8-byte words in the
/// order the thunk stored them.
pub(super) struct Done {
    outcome: Outcome,
    res: Vec<u64>,
}

enum TaskState {
    Running,
    Finished(Done),
    /// Moved into its group by a join.
    Taken,
}

/// One started call.
pub(super) struct Task {
    program: Arc<Program>,
    thunk: usize,
    depth: i32,
    /// The stack limit a run on the calling thread gets.
    limit: i64,
    site: i32,
    args: Vec<u64>,
    nres: usize,
    state: Mutex<TaskState>,
    finished: Condvar,
}

impl Task {
    /// Waits until the call has ended.
    fn wait(&self) {
        let mut s = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        while matches!(*s, TaskState::Running) {
            s = self
                .finished
                .wait(s)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn finish(&self, done: Done) {
        *self.state.lock().unwrap_or_else(PoisonError::into_inner) = TaskState::Finished(done);
        self.finished.notify_all();
    }

    /// Waits for the call, and takes what it left.
    fn take(&self) -> Option<Done> {
        self.wait();
        let mut s = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        match std::mem::replace(&mut *s, TaskState::Taken) {
            TaskState::Finished(d) => Some(d),
            _ => None,
        }
    }
}

/// The calls started by one window of a function, in program order.
#[derive(Default)]
pub(super) struct Group {
    tasks: Vec<Arc<Task>>,
    /// What each call left once a join resolved it.
    done: Vec<Option<Done>>,
}

/// Waits for every call of every group the run left open (a run that stopped
/// in a way that skipped a join), so that no worker is still running native
/// code when the run ends.
pub(super) fn wait_all(groups: &[Option<Group>]) {
    for g in groups.iter().flatten() {
        for t in &g.tasks {
            t.wait();
        }
    }
}

/// Calls `thunk`.
///
/// # Safety
///
/// `thunk` must be the address of a thunk generated by `module.rs` for the
/// program of the run (same signature), whose code is still alive; `args`
/// must have the words the thunk reads, and `res` room for the words it
/// writes.
unsafe fn call_thunk(
    thunk: usize,
    depth: i32,
    limit: i64,
    site: i32,
    args: &[u64],
    res: &mut [u64],
) -> i32 {
    // SAFETY: the caller promises `thunk` has this signature (`Thunk`).
    let f: Thunk = unsafe { std::mem::transmute::<usize, Thunk>(thunk) };
    // SAFETY: the caller promises the buffers are as long as the thunk needs.
    unsafe { f(depth, limit, site, args.as_ptr(), res.as_mut_ptr()) }
}

/// The outcome of the call that just ran on this thread, taken from its
/// runtime state (which is left clean).
fn take_outcome(st: &mut State, on_worker: bool) -> Outcome {
    let stack = std::mem::take(&mut st.stack_trip) && on_worker;
    if std::mem::take(&mut st.deopt) {
        st.error = None;
        return Outcome::Deopt;
    }
    match st.error.take() {
        Some(_) if stack => Outcome::Retry,
        Some(e) => Outcome::Err(e),
        None => Outcome::Err(internal("a failure without a diagnostic")),
    }
}

/// Runs the call on this thread, on this thread's runtime state (the state of
/// the caller). The thunk runs outside the state's borrow: the native code
/// it runs calls back into the runtime.
fn run_here(task: &Task) -> Done {
    let mut res = vec![0u64; task.nres];
    // SAFETY: `task.thunk` was made by `sky_rt_spawn` from an address native
    // code took of the thunk of this program, which is alive while the run
    // is (`Compiled::run` does not return before every group is joined or
    // waited for, and owns the code); `args` and `res` have the lengths
    // native code and `sky_rt_spawn` agreed on (`nargs`, `nres`).
    let status = unsafe {
        call_thunk(
            task.thunk, task.depth, task.limit, task.site, &task.args, &mut res,
        )
    };
    if status == 0 {
        return Done {
            outcome: Outcome::Ok,
            res,
        };
    }
    // On the calling thread a stack that ran out is the error the sequential
    // run reports.
    let outcome = with(Outcome::Err(internal("no runtime state")), |st| {
        take_outcome(st, false)
    });
    Done {
        outcome,
        res: Vec::new(),
    }
}

thread_local! {
    /// The kernels a worker compiled for the program it last ran, kept across
    /// calls (compiling a body costs about what a call of it costs). The
    /// program is held too, so the operations' addresses they are keyed by
    /// cannot be reused by another.
    static WORKER_KERNELS: std::cell::RefCell<Option<(Arc<Program>, Kernels)>> =
        const { std::cell::RefCell::new(None) };
}

/// Runs the call on a worker, on a runtime state of its own.
fn run_on_worker(task: Arc<Task>) {
    let ran = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| worker_body(&task)));
    let done = ran.unwrap_or_else(|_| Done {
        outcome: Outcome::Err(internal("a worker panicked")),
        res: Vec::new(),
    });
    task.finish(done);
}

fn worker_body(task: &Task) -> Done {
    crate::par::mark_call_worker();
    let kernels = WORKER_KERNELS
        .with(|k| k.borrow_mut().take())
        .and_then(|(p, k)| Arc::ptr_eq(&p, &task.program).then_some(k));
    // A started call prints nothing (its callee has no `io`).
    let mut sink = std::io::sink();
    let guard = begin_with(task.program.clone(), &mut sink, kernels);
    // Half of the run's budget: a call that needs more is run again on the
    // calling thread, which has the run's whole stack.
    let limit = stack_address().saturating_sub(STACK_BUDGET / 2) as i64;
    let mut res = vec![0u64; task.nres];
    // SAFETY: as in `run_here`; the worker's state is installed.
    let status = unsafe {
        call_thunk(
            task.thunk, task.depth, limit, task.site, &task.args, &mut res,
        )
    };
    let outcome = if status == 0 {
        Outcome::Ok
    } else {
        with(Outcome::Err(internal("no runtime state")), |st| {
            take_outcome(st, true)
        })
    };
    if let Some(kernels) = take_kernels() {
        WORKER_KERNELS.with(|k| *k.borrow_mut() = Some((task.program.clone(), kernels)));
    }
    drop(guard);
    Done {
        outcome,
        res: if status == 0 { res } else { Vec::new() },
    }
}

// ----- the calls native code makes -----

/// Opens a group: the handle of the list of calls a window starts.
#[no_mangle]
pub extern "C" fn sky_rt_group_begin() -> i64 {
    with(-1, |st| {
        let group = Some(Group::default());
        match st.groups.iter().position(Option::is_none) {
            Some(i) => {
                st.groups[i] = group;
                i as i64
            }
            None => {
                st.groups.push(group);
                st.groups.len() as i64 - 1
            }
        }
    })
}

/// Starts the call whose thunk is at `thunk`, with the `nargs` words at
/// `args` (copied), expecting `nres` words back, as the next call of group
/// `group`. `depth` is the callee's call depth and `limit` the caller's stack
/// limit.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn sky_rt_spawn(
    group: i64,
    thunk: i64,
    depth: i32,
    limit: i64,
    site: i32,
    args: i64,
    nargs: i32,
    nres: i32,
) {
    // SAFETY: native code passes the address of `nargs` words it just wrote
    // on its stack (a stack slot of at least one word).
    let args: Vec<u64> =
        unsafe { std::slice::from_raw_parts(args as *const u64, nargs.max(0) as usize) }.to_vec();
    let Some((program, threads)) = with(None, |st| {
        Some((st.program.clone(), st.program.par.threads))
    }) else {
        return;
    };
    // Another thread only while the caller is far from its stack budget, and
    // never from a worker (see the module docs).
    let start = (limit as usize).saturating_add(STACK_BUDGET);
    let used = start.saturating_sub(stack_address());
    let pool = if used <= STACK_BUDGET / 2 && !crate::par::on_call_worker() {
        crate::par::call_pool_for(threads)
    } else {
        None
    };
    let task = Arc::new(Task {
        program,
        thunk: thunk as usize,
        depth,
        limit,
        site,
        args,
        nres: nres.max(0) as usize,
        state: Mutex::new(TaskState::Running),
        finished: Condvar::new(),
    });
    crate::par::count_call(pool.is_some());
    match pool {
        Some(pool) => {
            let t = task.clone();
            pool.spawn(move || run_on_worker(t));
        }
        None => {
            let done = run_here(&task);
            task.finish(done);
        }
    }
    with((), |st| {
        if let Some(Some(g)) = usize::try_from(group)
            .ok()
            .and_then(|i| st.groups.get_mut(i))
        {
            g.tasks.push(task);
            g.done.push(None);
        } else {
            st.fail(internal("a call was started in a group that is not open"));
        }
    });
}

/// Waits for every call the group has started, and returns 0 if they all
/// succeeded. Otherwise it installs the outcome of the earliest failing one as
/// the run's pending error (replacing any: see the module docs) and returns
/// 1. Safe to call again: the calls started since are waited for, the outcome
/// is the same.
#[no_mangle]
pub extern "C" fn sky_rt_join(group: i64) -> i32 {
    let Some(tasks) = with(None, |st| {
        usize::try_from(group)
            .ok()
            .and_then(|i| st.groups.get(i))
            .and_then(Option::as_ref)
            .map(|g| g.tasks.clone())
    }) else {
        return 1;
    };
    for (k, task) in tasks.iter().enumerate() {
        let resolved = with(true, |st| {
            group_of(st, group).is_none_or(|g| g.done.get(k).is_some_and(Option::is_some))
        });
        if resolved {
            continue;
        }
        let mut done = task.take().unwrap_or(Done {
            outcome: Outcome::Err(internal("a started call left nothing")),
            res: Vec::new(),
        });
        if let Outcome::Retry = done.outcome {
            done = run_here(task);
        }
        with((), |st| {
            if let Some(slot) = group_of(st, group).and_then(|g| g.done.get_mut(k)) {
                *slot = Some(done);
            }
        });
    }
    with(1, |st| {
        let failure = group_of(st, group).and_then(|g| {
            g.done.iter().flatten().find_map(|d| match &d.outcome {
                Outcome::Ok => None,
                Outcome::Err(e) => Some(Some(e.clone())),
                Outcome::Deopt | Outcome::Retry => Some(None),
            })
        });
        match failure {
            None => 0,
            Some(Some(e)) => {
                st.error = Some(e);
                st.deopt = false;
                1
            }
            Some(None) => {
                st.error = None;
                st.deopt = true;
                1
            }
        }
    })
}

/// The words call `k` of the group left (after a join that returned 0).
#[no_mangle]
pub extern "C" fn sky_rt_task_result(group: i64, k: i32) -> i64 {
    with(0, |st| {
        group_of(st, group)
            .and_then(|g| g.done.get(usize::try_from(k).ok()?))
            .and_then(Option::as_ref)
            .map_or(0, |d| d.res.as_ptr() as i64)
    })
}

/// Ends the group.
#[no_mangle]
pub extern "C" fn sky_rt_group_end(group: i64) {
    with((), |st| {
        if let Some(slot) = usize::try_from(group)
            .ok()
            .and_then(|i| st.groups.get_mut(i))
        {
            *slot = None;
        }
    });
}

fn group_of(st: &mut State, group: i64) -> Option<&mut Group> {
    usize::try_from(group)
        .ok()
        .and_then(|i| st.groups.get_mut(i))
        .and_then(Option::as_mut)
}
