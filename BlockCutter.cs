using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using System.Threading;

namespace OrdererTest
{

    class BlockCutter
    {

        const ulong MaxPendingSizeBytes = 1048576;
        const int MaxPendingMsgCount = 10;

        public SafeBatch PendingBatch = new SafeBatch();
        ulong PendingBatchSizeBytes = 0;

        private bool pendinglocked = false; //Make sure multiple threads don't attempt to write to pending at the same time

        int MsgNum = 0;

        //Return and reset pending batch
        public SafeBatch Cut()
        {
            SafeBatch returnbatch = PendingBatch; //Set batch to return
            PendingBatch = new SafeBatch();
            PendingBatch.Messages = new SafeEnvelope[MaxPendingMsgCount];
            PendingBatch.MsgCount = 0;
            PendingBatchSizeBytes = 0;
            pendinglocked = false;
            return returnbatch;
        }

        //Add message to pending batch
        private void AddToPending(SafeEnvelope msg, ulong size, int index)
        {
            PendingBatch.Messages[PendingBatch.MsgCount] = msg;
            Interlocked.Increment(ref PendingBatch.MsgCount);
            PendingBatchSizeBytes += size;
        }


        //Will return 0, 1, or 2 batches
        public SafeOrderedReturn Ordered(SafeEnvelope msg, int index)
        {

            try
            {
                SafeBatch[] batches = new SafeBatch[2];
                bool ispendingbatch = true;
                ulong MsgSizeBytes = (ulong)msg.Payload.Length; //Length of byte array = size in bytes

                //Message is too big and thus will overflow, send pending and this msg in its own batch
                if (MsgSizeBytes > MaxPendingSizeBytes)
                {
                    if (PendingBatch.Messages.Count() > 0)
                    {
                        batches[0] = Cut(); //Add pending batch to return
                    }
                    SafeBatch newbatch = new SafeBatch();
                    newbatch.Messages = new SafeEnvelope[1];
                    newbatch.Messages[0] = msg;
                    batches[1] = newbatch;
                    ispendingbatch = false;
                }
                else
                {
                    //Message will cause overflow, cut pending batch
                    if (PendingBatchSizeBytes + MsgSizeBytes > MaxPendingSizeBytes)
                    {
                        batches[0] = Cut(); //Add pending batch to return
                    }

                    while (index > MsgNum || pendinglocked) { } //Wait for it to be this msg's turn and not currently cutting batch

                    AddToPending(msg, MsgSizeBytes, index); //Add message to pending batch

                    //Pending batch has reached max count, must cut
                    //If the last if statement was true, this one will not be true
                    if (PendingBatch.MsgCount >= MaxPendingMsgCount)
                    {
                        pendinglocked = true;
                        batches[1] = Cut(); //Add pending batch to return
                        ispendingbatch = false;
                    }

                    Interlocked.Add(ref MsgNum, 1); //Increment msgnum
                }
                SafeOrderedReturn toreturn = new SafeOrderedReturn();
                toreturn.Batches = batches;
                toreturn.IsPendingBatch = ispendingbatch;
                return toreturn;
            }
            catch
            {
                int a = 1;
                return new SafeOrderedReturn();
            }



        }

        public BlockCutter()
        {
            PendingBatch.Messages = new SafeEnvelope[MaxPendingMsgCount];
            PendingBatch.MsgCount = 0;
        }

    }
}
