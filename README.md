# OrdererTest

## Background

First the background, but this part isn't too important to understand the code, just why I'm writing it. Hyperledger Fabric is a private blockchain mainly intended for organizations to be able to update a shared database. When an update is made to a database, such as a transaction, that information is stored in a message. The "ordering service" packages these messages into a batch which eventually becomes one block. (The payload portion of a message are just bytes representing the update.) The ordering service currently in use is single threaded and has no parallelism. 

The goal of my research is to create a highly parallelized and multithreaded version of the ordering service using the GPU. Just for reference, the main code of the ordering service (although, for me, Go was hard to understand until I actually learned it since it doesn't look like any other language)

[Ordering service](https://github.com/hyperledger/fabric/blob/3beba719198d9b34dcf2e0b2e88c4072d4a58a0f/orderer/consensus/solo/consensus.go#L103) (main function is the important one)

[Used by the ordering service](https://github.com/hyperledger/fabric/blob/3beba719198d9b34dcf2e0b2e88c4072d4a58a0f/orderer/common/blockcutter/blockcutter.go#L69) (Ordered function)

Basically, messages come in, and are added to a pending batch. Once some condition has been fulfilled, such as reaching the max number of pending messages, the pending batch will be "cut" and made into a block. At first I tried to create everything in C, but I found some of the higher level constructs (for example, JSON) were hard to use with C. So instead I have the main part of the service made with C#, but use the GPU for keeping track of the pending batch.

## Program Explanation/Walkthrough

Now I'll attempt to walk through what the code does, starting with the main function on [line 223 in Program.cs](Program.cs#L223). I'm not sure if you've used C#, but hopefully its syntax is mostly familiar. The main function simulates a bunch of messages coming in, where each message is given a thread (created with Task.Factory.StartNew()) that runs the function [HandleMessage on line 49](Program.cs#L49). This function first checks if the message is a config message or not since those are handled differently, but for my current testing none are. It then creates two empty batches passed by reference to the function in [kernel.cu Ordered on line 275](kernel.cu#L275). kernel.cu is compiled as orderer_kernel.dll and this is how Program.cs references it.

Ordered does some small memory management and then calls [OrderedGPU, line 215](kernel.cu#L215). This is where the different possible conditions are checked for, but for my current testing, the condition code will always either be 0 (no need to cut the pending batch) or 4 (exceeded max msg count, need to cut the pending batch.) Either way, it sends the message to [AddToPendingGPU (line 166)](kernel.cu#L166) which simply adds the message to the current pending batch. There are two pending batches, which one used is decided  by the WhichBatch bool, which allows one batch to have its data copied while modifying the other one. Going back to Ordered, it checks the condition code set in OrderedGPU to decide whether the batch needs to be cut or not. If it does, it calls [Cut (line 123)](kernel.cu#L123). Cut performs the necessary memory management before calling [CutGPU on line 105](kernel.cu#L105) in order to extract all the messages from the current pending batch. CutGPU also toggles WhichBatch and resets the batch being copies from. Finally, Ordered returns a boolean which indicates whether or not there is currently a pending batch.

Back in [Program.cs](Program.cs#L76), both batches are checked to see if they have more than 0 messages (currently, only Batch2 possibly will) and if so they are dispatched in a new thread to be made into a block. Some handling is then done for a timer, which ensures that at least one block will be created every x seconds as long as there are messages waiting, but this part isn't important.

In main, once all messages have been sent to their own respective threads, it waits for the number of blocks to equal what it should be, indicating everything has finished running, and then outputs some timing information. You can see there is a high level of threads being created both on the CPU and GPU which can make debugging difficult. Everything works if I make the code more inefficient so that there are less threads running at once concurrently. 

Note that there are two other files I didn't describe here: [BlockCutter.cs](BlockCutter.cs) (only used by the CPU version of my program when useGPU is set to false, which is not at issue here, although it is slow) and [BlockWriter.cs](BlockWriter.cs) (used for block creation by both GPU and CPU version, but again, is not at issue here so I didn't include it). Also, the reason for things named "Safe" or "Unsafe" is because C# considers anything that uses pointers to be unsafe. You normally don't need pointers in C#, but I have to use them so that the data structures match those in kernel.cu exactly or else the dll call will error.

## Current issues

The main issues are with concurrency in the GPU threads. The pendingbatch structures need to be ensured that they're modified in the correct order. For that, I have this line in AddToPendingGPU:

    while (index > MsgNum || BatchFull) __nanosleep(100); //Wait until msg index equals next msg num and batch doesn't need to be cut
    
Index is provided by the C# program in the order the messages come in, and MsgNum is incremented every message, ensuring that they're ordered correctly. BatchFull indicates the current batch needs to be cut and WhichBatch must be toggled before it can add the message. The issue is that whenever one thread gets to this waiting loop, all other threads stall as well. This can be seen by uncommenting the print statements on lines [168](kernel.cu#L168) and [277](kernel.cu#L277) and inspecting the output after sending 10 messages in. Here is such an output:

```
Ordered start: 0
Index: 0, MsgNum: 0
Ordered start: 1
Index: 1, MsgNum: 1
Ordered start: 2
Index: 2, MsgNum: 2
Ordered start: 6
Ordered start: 8
Ordered start: 5
Ordered start: 7
Ordered start: 9
Ordered start: 4
Ordered start: 3
Index: 9, MsgNum: 3 
```

The program then hangs indefinitely when this while loop is reached. I asked this problem to Dr. Koppelman. His suggestion was that all GPU threads were using the default stream, which meant they would all stall if any one did. The suggested solution is either to add the flag ```--default-stream per-thread``` during compilation in order to make each thread have its own stream, or to manually create a new stream in the Ordered function before performing the kernel launch. Attempts at the latter option can be seen commented out from line [307-323](kernel.cu#L307). Either way, the same thing as above happened when trying these options.

I believe there may be some other concurrency issues with inspecting values of variables like PendingBatchSizeBytes in OrderedGPU before they're acted upon in Ordered. Besides that, everything seems to work correctly: If I run the program linearly rather than multiple concurrent threads, everything works as expected.

## Other info

I am on Windows 10, an RTX 2070 Super, Cuda 11.0.3, and the program is compiled using Visual Studio. The following two commands are used to compile and link kernel.cu:

    “C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin\nvcc.exe” -gencode=arch=compute_70,code=“sm_70,compute_70” --use-local-env -ccbin “C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.24.28314\bin\HostX86\x64” -x cu -rdc=true -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\include" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\include" --keep-dir x64\Release -maxrregcount=0 --machine 64 --compile -cudart static --default-stream per-thread -use_fast_math -DWIN32 -DWIN64 -DNDEBUG -D_CONSOLE -D_WINDLL -D_MBCS -Xcompiler “/EHsc /W3 /nologo /Ox /Fdx64\Release\vc142.pdb /FS /Zi /MD " -o x64\Release\kernel.cu.obj “orderer_kernel\kernel.cu”
    
    “C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin\nvcc.exe” -dlink -o x64\Release\orderer_kernel.device-link.obj -Xcompiler “/EHsc /W3 /nologo /Ox /Zi /Fdx64\Release\vc142.pdb /MD " -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin/crt” -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib\x64” cudart_static.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib cudart.lib cudadevrt.lib --default-stream per-thread -gencode=arch=compute_70,code=sm_70 --machine 64 x64\Release\kernel.cu.obj
