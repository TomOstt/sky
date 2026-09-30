// Naive doubly recursive Fibonacci. usage: fib N
fn fib(n: u64) -> u64 {
    if n < 2 {
        n
    } else {
        fib(n - 1) + fib(n - 2)
    }
}

fn main() {
    let n: u64 = std::env::args().nth(1).map_or(30, |a| a.parse().unwrap());
    println!("{}", fib(n));
}
