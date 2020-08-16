#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <vector>
#include <device_atomic_functions.h>
#include <atomic>

struct Envelope
{
    char* Payload;
    char* Signature;
};

struct Batch
{
    Envelope* Messages;
    int MsgCount;
};

#define __CUDACC__

__device__ const long MaxPendingSizeBytes = 1048576;
__device__ const int MaxPendingMsgCount = 10;
__device__ const int MaxMsgSize = 10000; //Message can be this many chars, maybe need bigger?

//Pending batch structures hold messages being ordered until it's time to transfer them out
__device__ Batch PendingBatch1;
__device__ Batch PendingBatch2;
__device__ int WhichBatch = false; //False = PendingBatch1, true = PendingBatch2
__device__ long PendingBatchSizeBytes = 0;
__device__ int BatchFull = false;
__device__ int MsgNum = 0;

cudaStream_t streams[100];

__global__ void Init()
{
    PendingBatch1.Messages = new Envelope[MaxPendingMsgCount];
    PendingBatch1.MsgCount = 0;
    PendingBatch2.Messages = new Envelope[MaxPendingMsgCount];
    PendingBatch2.MsgCount = 0;
    ////For testing
    //Envelope env1;
    //env1.Payload = "BQeUTS2rn9uC/XfzOus3aQ==";
    //env1.Signature = "/ULc6mTCl/oZlCJa4OkD0w==";
    //Envelope env2;
    //env2.Payload = "KMGirCuwYl8HGNuVjz3vPw==";
    //env2.Signature = "kyWJOTnbokNvmTOZiEX6mQ==";
    //PendingBatch1.Messages[0] = env1;
    //PendingBatch1.Messages[1] = env2;
    //PendingBatch1.MsgCount = 2;
}

__global__ void GetPendingCountGPU(int* pendingcnt)
{
    *pendingcnt = WhichBatch ? PendingBatch2.MsgCount : PendingBatch1.MsgCount;
}

extern "C" int __declspec(dllexport) __stdcall GetPendingCount()
{

    int* pendingcnt_d;
    cudaMalloc(&pendingcnt_d, sizeof(int));
    int pendingcnt_h = 0;

    cudaMemcpy(pendingcnt_d, &pendingcnt_h, sizeof(pendingcnt_h), cudaMemcpyHostToDevice);

    GetPendingCountGPU<<<1, 1>>>(pendingcnt_d);

    cudaMemcpy(&pendingcnt_h, pendingcnt_d, sizeof(pendingcnt_h), cudaMemcpyDeviceToHost);

    return pendingcnt_h;

    ////Supposed to be able to use cudaMemcpyFromSymbol to do this easier, but I kept getting error symbol unrecognized
    //int* cnt = new int;
    //cudaError_t cudaStatus = cudaMemcpyFromSymbol(cnt, &PendingBatch.MsgCount, sizeof(int), 0, cudaMemcpyHostToDevice);
    //if (cudaStatus != cudaSuccess)
    //{
    //    int a = 1;
    //}
    //return *cnt;

}

__device__ char* strcpygpu(char* dest, const char* src) {
    int i = 0;
    do
    {
        dest[i] = src[i];
    } while (src[i++] != 0);
    return dest;
}

//Transfers byte strings from pending batch to output batch
__global__ void TransferStrings(Batch* b)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x; //Which message it is
    strcpygpu(b->Messages[i].Payload, WhichBatch ? PendingBatch1.Messages[i].Payload : PendingBatch2.Messages[i].Payload); //Which batch reversed since WhichBatch is toggled before running this
    strcpygpu(b->Messages[i].Signature, WhichBatch ? PendingBatch1.Messages[i].Signature : PendingBatch2.Messages[i].Signature); //Which batch reversed since WhichBatch is toggled before running this
}

//Transfers messages in pending batch to b, resets that pending batch, and 
__global__ void CutGPU(Batch* b)
{

    atomicExch(&WhichBatch, !WhichBatch); //Toggle which batch to use
    PendingBatchSizeBytes = 0; //Can't use atomicexch since type is long, maybe worth it to switch to int

    atomicExch(&BatchFull, false); //Now that WhichBatch has been toggled, can handle the previous batch while modifying the current one

    //Transfer pending batch to caller, reset pending batch
    TransferStrings<<<1, WhichBatch ? PendingBatch1.MsgCount : PendingBatch2.MsgCount>>>(b);

    b->MsgCount = WhichBatch ? PendingBatch1.MsgCount : PendingBatch2.MsgCount; //Which batch reversed since WhichBatch is toggled before running this

    atomicExch(WhichBatch ? &PendingBatch1.MsgCount : &PendingBatch2.MsgCount, 0); //Set msgcount to 0 for correct batch

}

//Retrieves the current pending batch
extern "C" Batch __declspec(dllexport) __stdcall Cut()
{
    //Batch allocation
    Batch* batch_d;
    cudaMalloc(&batch_d, sizeof(*batch_d));
    Batch batch_h;
    //Envelope allocation
    Envelope* mess_d; //Array of size MaxPendingMsgCount
    cudaMalloc(&mess_d, sizeof(*mess_d) * MaxPendingMsgCount);
    batch_h.Messages = mess_d;
    //Strings allocation
    std::vector<Envelope> msgs(MaxPendingMsgCount);
    for (auto& m : msgs)
    {
        cudaMalloc(&m.Payload, MaxMsgSize);
        cudaMalloc(&m.Signature, MaxMsgSize);
    }
    cudaMemcpy(mess_d, msgs.data(), msgs.size() * sizeof(msgs[0]), cudaMemcpyHostToDevice);
    //Copy to GPU
    cudaMemcpy(batch_d, &batch_h, sizeof(*batch_d), cudaMemcpyHostToDevice);
    //Run GPU function
    CutGPU<<<1, 1>>>(batch_d);
    //Batch back to CPU
    cudaMemcpy(&batch_h, batch_d, sizeof(*batch_d), cudaMemcpyDeviceToHost);
    //Messages back to CPU
    batch_h.Messages = new Envelope[MaxPendingMsgCount];
    cudaMemcpy(batch_h.Messages, mess_d, sizeof(*mess_d) * MaxPendingMsgCount, cudaMemcpyDeviceToHost);
    //Strings back to CPU
    for (int i = 0; i < batch_h.MsgCount; i++)
    {
        Envelope& eh = batch_h.Messages[i];
        eh = { (char*)malloc(MaxMsgSize), (char*)malloc(MaxMsgSize) };

        Envelope& ed = msgs[i]; // This has device pointers.

        cudaMemcpy(eh.Payload, ed.Payload, MaxMsgSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(eh.Signature, ed.Signature, MaxMsgSize, cudaMemcpyDeviceToHost);
    }
    return batch_h;
    //*batch = batch_h;
}

//Adds msg to pending batch in GPU
__global__ void AddToPendingGPU(Envelope* msg, long len, int index)
{
    //printf("Index: %i, MsgNum: %i\n", index, MsgNum);

    while (index > MsgNum || BatchFull) __nanosleep(100); //Wait until msg index equals next msg num and batch doesn't need to be cut

    if (WhichBatch)
    {
        PendingBatch2.Messages[atomicAdd(&PendingBatch2.MsgCount, 1)] = *msg; //Add message to pending batch, and increment msgcount
    }
    else
    {
        PendingBatch1.Messages[atomicAdd(&PendingBatch1.MsgCount, 1)] = *msg; //Add message to pending batch, and increment msgcount
    }
    PendingBatchSizeBytes += len;
    atomicAdd(&MsgNum, 1);
}

//Adds msg to pending batch from CPU call
void AddToPending(Envelope msg, int index)
{
    //Memory for msg

    //Lengths and initial device and host vars
    long len = strlen(msg.Payload);
    long siglen = strlen(msg.Signature);
    Envelope* msg_d;
    cudaMalloc(&msg_d, sizeof(*msg_d));
    Envelope msg_h;
    //Payload allocate and copy
    char* msgpayload_d;
    cudaMalloc(&msgpayload_d, len);
    msg_h.Payload = msgpayload_d;
    cudaMemcpy(msgpayload_d, msg.Payload, len, cudaMemcpyHostToDevice);
    //Signature allocate and copy
    char* msgsig_d;
    cudaMalloc(&msgsig_d, siglen);
    msg_h.Signature = msgsig_d;
    cudaMemcpy(msgsig_d, msg.Signature, siglen, cudaMemcpyHostToDevice);
    //Copy everything to device
    cudaMemcpy(msg_d, &msg_h, sizeof(*msg_d), cudaMemcpyHostToDevice);

    AddToPendingGPU<<<1, 1>>>(msg_d, len, index); //Call GPU function
}

//Condition denotes the situation caused by observing/adding the message and the current batch
//Condition is 0 by default, means do nothing
//If condition is 1 or 2, then conditions 3 or 4 can't be true
//If condition is 3, then condition 4 can't be true
__global__ void OrderedGPU(Envelope* msg, long len, int* ConditionCode, int index)
{
    //printf("Which batch: %i\n", WhichBatch);

    *ConditionCode = 0;

    //Message is too big and thus will overflow, send pending and this msg in its own batch
    if (len > MaxPendingSizeBytes)
    {
        //CONDITION 1, need to cut pending batch and isolate this msg
        if (WhichBatch ? PendingBatch2.MsgCount : PendingBatch1.MsgCount > 0)
        {
            *ConditionCode = 1;
        }
        //CONDITION 2, no need to cut pendign batch, still isolate this msg
        else
        {
            *ConditionCode = 2;
        }
        //printf("Hello World from GPU! %s\n", Batch2->Messages[0].Payload);
    }
    else
    {
        //CONDITION 3, Message will cause overflow, cut pending batch
        if (PendingBatchSizeBytes + len > MaxPendingSizeBytes)
        {
            *ConditionCode = 3;
            //Message still needs to be added, but it will be called manually from CPU
        }
        else //At this point condition code equals either 0 or 4
        {
            AddToPendingGPU<<<1, 1>>>(msg, len, index);

            cudaDeviceSynchronize();

            //printf("PB1 cnt: %i\n", PendingBatch1.MsgCount);
            //printf("PB2 cnt: %i\n", PendingBatch2.MsgCount);

            //CONDITION 4 (only one that should occur during speed testing)
            //Pending batch has reached max count, must cut
            //If the last if statement was true, this one will not be true
            if((WhichBatch ? PendingBatch2.MsgCount : PendingBatch1.MsgCount) >= MaxPendingMsgCount)
            {
                //printf("Batch cut\n");
                atomicExch(&BatchFull, true);
                *ConditionCode = 4;
            }
        }
    }
}

//Transfers msg into the OrderedGPU funtion which returns a condition code
//If msg is isolated in its own batch, Batch1 will be used. If a batch is cut, Batch2 will be used.
//Bool return value indicates if there is a pending batch. Batch1 and Batch2 are also written to and returned.
//Condition 0: Do nothing, msg just added to pending in ordered
//Condition 1: Must cut pending batch, and isolate msg in its own batch. The latter can easily be accomplished on the CPU.
//Condition 2: Don't cut pending batch, sitll isolate msg in its own batch. 
//Condition 3: Cut pending batch, then add msg. This means msg won't be added in OrderedGPU, and will have to have a separate kernel launch to add after cutting in this function.
//Condition 4: Cut pending batch after adding msg. This is the simplest one to handle.
//Also, there will be a pending batch after execution if condition is 1, 2, or 4.
extern "C" bool __declspec(dllexport) __stdcall Ordered(Envelope msg, Batch * Batch1, Batch * Batch2, int index)
{
    //printf("Ordered start: %i\n", index);

    //Memory for msg

    //Lengths and initial device and host vars
    long len = strlen(msg.Payload);
    long siglen = strlen(msg.Signature);
    Envelope* msg_d;
    cudaMalloc(&msg_d, sizeof(*msg_d));
    Envelope msg_h;

    //Payload allocate and copy
    char* msgpayload_d;
    cudaMalloc(&msgpayload_d, len);
    msg_h.Payload = msgpayload_d;
    cudaMemcpy(msgpayload_d, msg.Payload, len, cudaMemcpyHostToDevice);

    //Signature allocate and copy
    char* msgsig_d;
    cudaMalloc(&msgsig_d, siglen);
    msg_h.Signature = msgsig_d;
    cudaMemcpy(msgsig_d, msg.Signature, siglen, cudaMemcpyHostToDevice);
    //Copy everything to device
    cudaMemcpy(msg_d, &msg_h, sizeof(*msg_d), cudaMemcpyHostToDevice);
    //Int for the condition code
    int* ConditionCode_d;
    cudaMalloc(&ConditionCode_d, sizeof(int));
    int ConditionCode_h = 0;
    cudaMemcpy(ConditionCode_d, &ConditionCode_h, sizeof(int), cudaMemcpyHostToDevice);

    //////////Streams test. Currently has same result as using default stream

    //cudaStreamCreate(&streams[index]);

    //cudaStream_t test;

    //cudaStreamCreateWithFlags(&test, cudaStreamNonBlocking);

    ///******************Run GPU function******************/
    //OrderedGPU<<<1, 1, 0, test>>>(msg_d, len, ConditionCode_d, index);


    ////Copy condition code back to CPU
    //cudaMemcpyAsync(&ConditionCode_h, ConditionCode_d, sizeof(int), cudaMemcpyDeviceToHost, test);


    //////////Normal test

    /******************Run GPU function******************/
    OrderedGPU<<<1, 1>>>(msg_d, len, ConditionCode_d, index);

    //Copy condition code back to CPU
    cudaMemcpy(&ConditionCode_h, ConditionCode_d, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(&msg_h);
    cudaFree(ConditionCode_d);

    //Handle based on condition code
    //Doing it this way avoids unnecessarily copying batches in/out the GPU
    Batch1->MsgCount = 0; //Init to 0
    Batch2->MsgCount = 0;
    if (ConditionCode_h > 0) //Need to handle something
    {
        if (ConditionCode_h < 3) //Need to isolate msg in batch1
        {
            Batch1->Messages[0] = msg;
            Batch1->MsgCount = 1;
            if (ConditionCode_h == 1) //Also need to cut pending batch
            {
                *Batch2 = Cut();
            }
        }
        else //Condition code equals 3 or 4
        {
            *Batch2 = Cut();
            if (ConditionCode_h == 3) //Also need to add msg to pending after cutting
            {
                AddToPending(msg, index);
            }
        }
    }
    return ConditionCode_h == 0 || ConditionCode_h == 3;
}

extern "C" int __declspec(dllexport) __stdcall main()
{
    Init<<<1, 1>>>();

    ////Declaring randomly generated envelopes
    //Envelope env1;
    //env1.Payload = "BQeUTS2rn9uC/XfzOus3aQ==";
    //env1.Signature = "/ULc6mTCl/oZlCJa4OkD0w==";
    //Envelope env2;
    //env2.Payload = "KMGirCuwYl8HGNuVjz3vPw==";
    //env2.Signature = "kyWJOTnbokNvmTOZiEX6mQ==";
    //Envelope env3;
    //env3.Payload = "+qPLeEmLPkzffONM5wJr3A==";
    //env3.Signature = "pOPGu1/KmUweXz2RfcjXwQ==";

    /**********Ordered testing**********/
    //printf("Pending cnt: %i\n", GetPendingCount());

    /**********Ordered testing**********/
    //Envelope msg = env3;
    //Batch* b1 = new Batch();
    //Batch* b2 = new Batch();
    //clock_t start_t = clock();
    //bool ispending = Ordered(msg, b1, b2, 2);
    //printf("Hello: %i\n", b2->MsgCount);
    //clock_t end_t = clock();
    //double ticks = (double)(end_t - start_t);
    //printf("Time taken: %f\n", ticks);

    //////Cont.
    //msg = env1;
    //b1 = new Batch();
    //b2 = new Batch();
    //ispending = Ordered(msg, b1, b2, 3);
    //printf("Hello: %i\n", b2->MsgCount);

    //msg = env2;
    //b1 = new Batch();
    //b2 = new Batch();
    //ispending = Ordered(msg, b1, b2, 4);
    //printf("Hello: %i\n", b2->MsgCount);


    /**********Cut testing**********/
    //clock_t start_t = clock();
    //Batch test = Cut();
    //printf("Hello: %s\n", test.Messages[2].Payload);
    //clock_t end_t = clock();
    //double ticks = (double)(end_t - start_t);
    //printf("Time taken: %f\n", ticks);

    return 0;
}