//  An example of stackful coroutine where each coroutine (or thread, as we call it in the example implementation) has its own stack. This also means that we can interrupt and resume execution at any point in time. It doesn’t matter if we’re in the middle of a stack frame (in the middle of executing a function); we can simply tell the CPU to save the state we need to the stack, return to a different stack and restore the state it needs there, and resume as if nothing has happened.


#![feature(naked_functions)]

use std::arch::asm;
const DEFAULT_STACK_SIZE: usize = 1024 * 1024 * 2; // 2MB
const MAX_THREADS: usize = 4;
static mut RUNTIME: usize = 0; // pointer to our runtime

pub struct Runtime {
    threads: Vec<Thread>,
    current: usize,
}

#[derive(PartialEq, Eq, Debug)]
enum State {
    Available, // Threads available to use
    Running,
    Ready, // we have work to do, thread is ready to do it
}

pub struct Thread {
    stack: Vec<u8>,
    // a context representing the data our CPU needs to resume where it left off on a stack.
    // ThreadContext holds data for the registers that the CPU needs to resume execution on a stack.
    ctx: ThreadContext,
    // thread state
    state: State,
}

#[derive(Debug, Default)]
#[repr(C)]
struct ThreadContext {
    rsp: u64, // 8 bytes
    r15: u64,
    r14: u64,
    r13: u64,
    r12: u64,
    rbx: u64,
    rbp: u64,
}

impl Thread {
    fn new() -> Self {
        Self {
            // not an optimal use of our resources to allocate stack size, but
            // lowers complexity of our code
            // side note: If the stack is reallocated, any pointers that we hold to it are invalidated. It’s worth mentioning that Vec<T> has a method called into_boxed_slice(), which returns a reference to an allocated slice Box<[T]>. Slices can’t grow, so if we store that instead, we can avoid the reallocation problem. There are several other ways to make this safer, but we’ll not focus on those in this example.
            stack: vec![0u8; DEFAULT_STACK_SIZE],
            ctx: ThreadContext::default(),
            state: State::Available,
        }
    }
}

impl Runtime {
    pub fn new() -> Self {
        let base_thread = Thread {
            stack: vec![0u8; DEFAULT_STACK_SIZE],
            ctx: ThreadContext::default(),
            state: State::Running,
        };
        let mut threads = vec![base_thread];
        let mut available_threads: Vec<Thread> = (1..MAX_THREADS).map(|_| Thread::new()).collect();

        // base_thread is state::Available, while all other threads are state::Running
        threads.append(&mut available_threads);

        Self {
            threads,
            current: 0,
        }
    }
    pub fn init(&self) {
        // we want to access our runtime struct from anywhere in our code so that we can call yield on it at any point in our code.
        // not the safest method to do it, but
        // again, to make this code less complex we do this.
        unsafe {
            let r_ptr: *const Runtime = self;
            RUNTIME = r_ptr as usize;
        }
    }

    // start running our runtime. It will continually call t_yield() until it returns false, which means that there is no more work to do and we can exit the process
    pub fn run(&mut self) -> ! {
        while self.t_yield() {}
        std::process::exit(0);
    }

    // function that we call when a thread is finished
    // Note: the user of our threads does not call this; we set up our stack so this is called when the task is done
    fn t_return(&mut self) {
        // If the calling thread is the base_thread, we won’t do anything
        if self.current != 0 {
            // We set its state to Available, letting the runtime know it’s ready to be assigned a new task
            self.threads[self.current].state = State::Available;

            // and then immediately call t_yield, which will schedule a new thread to be run.
            self.t_yield();
        }
    }
    // heart of our Runtime
    #[inline(never)]
    fn t_yield(&mut self) -> bool {
        let mut pos = self.current;
        while self.threads[pos].state != State::Ready {
            pos += 1;

            // wrap around the array of threads
            if pos == self.threads.len() {
                pos = 0;
            }

            if pos == self.current {
                return false;
            }
        }
        if self.threads[self.current].state != State::Available {
            self.threads[self.current].state = State::Ready;
        }
        self.threads[pos].state = State::Running;
        let old_pos = self.current;
        self.current = pos;
        unsafe {
            let old: *mut ThreadContext = &mut self.threads[old_pos].ctx;
            let new: *mut ThreadContext = &mut self.threads[pos].ctx;

            asm!(
                "call switch",
                in("rdi") old,
                in("rsi") new,
                clobber_abi("C")
            );
        }
        self.threads.len() > 0
    }
    pub fn spawn(&mut self, f: fn()) {
        let available = self
            .threads
            .iter_mut()
            .find(|t| t.state == State::Available)
            .expect("No available thread"); // panic if no available threads found.

        let size = available.stack.len();
        unsafe {
            let s_ptr = available.stack.as_mut_ptr().offset(size as isize);
            let s_ptr = (s_ptr as usize & !15) as *mut u8;
            std::ptr::write(s_ptr.offset(-16) as *mut u64, guard as u64);
            std::ptr::write(s_ptr.offset(-24) as *mut u64, skip as u64);
            std::ptr::write(s_ptr.offset(-32) as *mut u64, f as u64);
            available.ctx.rsp = s_ptr.offset(-32) as u64;
        }
        available.state = State::Ready;
    }
}

// The guard function is called when the function that we passed in, f, has returned. When f returns, it means our task is finished, so we de-reference our Runtime and call t_return()
fn guard() {
    unsafe {
        let rt_ptr = RUNTIME as *mut Runtime;
        (*rt_ptr).t_return();
    }
}

#[naked]
// ret will just pop off the next value from the stack and jump to whatever instructions that address points to. In our case, this is the guard function
unsafe extern "C" fn skip() {
    asm!("ret", options(noreturn))
}

// This helper function lets us call t_yield on our Runtime from an arbitrary place in our code without needing any references to it.
// This function is very unsafe, however making this safer is not a priority for us just to get our example up and running
pub fn yield_thread() {
    unsafe {
        let rt_ptr = RUNTIME as *mut Runtime;
        (*rt_ptr).t_yield();
    }
}

#[naked]
#[no_mangle]
unsafe extern "C" fn switch() {
    // to save and resume the execution.
    // first read out the values of all the registers we need and then set all the register values to the register values we saved when we suspended execution on the new thread.
    asm!(
        "mov [rdi + 0x00], rsp", // offset the pointer in 8-byte steps, same size as the u64 fields on our ThreadContext struct
        "mov [rdi + 0x08], r15",
        "mov [rdi + 0x10], r14",
        "mov [rdi + 0x18], r13",
        "mov [rdi + 0x20], r12",
        "mov [rdi + 0x28], rbx",
        "mov [rdi + 0x30], rbp",
        "mov rsp, [rsi + 0x00]",
        "mov r15, [rsi + 0x08]",
        "mov r14, [rsi + 0x10]",
        "mov r13, [rsi + 0x18]",
        "mov r12, [rsi + 0x20]",
        "mov rbx, [rsi + 0x28]",
        "mov rbp, [rsi + 0x30]",
        "ret",
        options(noreturn)
    );
}

fn main() {
    let mut runtime = Runtime::new();
    runtime.init();
    runtime.spawn(|| {
        println!("Thread 1 starting");
        let id = 1;
        for i in 0..10 {
            println!("thread: {} counter: {}", id, i);
            yield_thread();
        }
        println!("Thread 1 finished.");
    });
    runtime.spawn(|| {
        println!("Thread 2 starting.");
        let id = 2;
        for i in 0..15 {
            println!("thread: {} counter: {}", id, i);
            yield_thread();
        }
        println!("Thread 2 finished.");
    });
    runtime.run();
}
