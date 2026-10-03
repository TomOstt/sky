// R4: count Metal compute dispatches / barriers issued by this process.
// Swizzles the concrete MTLComputeCommandEncoder class of the default device.
// Build: clang -dynamiclib -fobjc-arc -framework Metal -framework Foundation -o libr4count.dylib r4count.m
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <objc/runtime.h>

static long g_tg = 0, g_th = 0, g_bar = 0, g_bar_res = 0, g_enc = 0;
static IMP o_tg, o_th, o_bar, o_bar_res, o_end;

static void my_tg(id s, SEL c, MTLSize a, MTLSize b) {
  g_tg++;
  ((void (*)(id, SEL, MTLSize, MTLSize))o_tg)(s, c, a, b);
}
static void my_th(id s, SEL c, MTLSize a, MTLSize b) {
  g_th++;
  ((void (*)(id, SEL, MTLSize, MTLSize))o_th)(s, c, a, b);
}
static void my_bar(id s, SEL c, MTLBarrierScope sc) {
  g_bar++;
  ((void (*)(id, SEL, MTLBarrierScope))o_bar)(s, c, sc);
}
static void my_bar_res(id s, SEL c, const id* r, NSUInteger n) {
  g_bar_res++;
  ((void (*)(id, SEL, const id*, NSUInteger))o_bar_res)(s, c, r, n);
}
static void my_end(id s, SEL c) {
  g_enc++;
  ((void (*)(id, SEL))o_end)(s, c);
}

static IMP swz(Class cls, SEL sel, IMP imp) {
  Method m = class_getInstanceMethod(cls, sel);
  if (!m) return NULL;
  return method_setImplementation(m, imp);
}

int r4_install(void) {
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc =
        [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    Class cls = object_getClass(enc);
    [enc endEncoding];
    o_tg = swz(cls, @selector(dispatchThreadgroups:threadsPerThreadgroup:), (IMP)my_tg);
    o_th = swz(cls, @selector(dispatchThreads:threadsPerThreadgroup:), (IMP)my_th);
    o_bar = swz(cls, @selector(memoryBarrierWithScope:), (IMP)my_bar);
    o_bar_res = swz(cls, @selector(memoryBarrierWithResources:count:), (IMP)my_bar_res);
    o_end = swz(cls, @selector(endEncoding), (IMP)my_end);
    NSLog(@"r4count: swizzled %s tg=%p th=%p bar=%p", class_getName(cls), o_tg, o_th, o_bar);
    return (o_tg != NULL) + 2 * (o_th != NULL) + 4 * (o_bar != NULL);
  }
}

void r4_counts(long* out) {
  out[0] = g_tg; out[1] = g_th; out[2] = g_bar; out[3] = g_bar_res; out[4] = g_enc;
}
void r4_reset(void) { g_tg = g_th = g_bar = g_bar_res = g_enc = 0; }
