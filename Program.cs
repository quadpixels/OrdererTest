using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using System.IO;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;

namespace OrdererTest
{

    class GPUWrapper
    {
        [DllImport("orderer_kernel.dll", CharSet = CharSet.Auto)]
        public static extern int GetPendingCount();

        [DllImport("orderer_kernel.dll", CharSet = CharSet.Auto)]
        public static extern Batch Cut();

        [DllImport("orderer_kernel.dll", CharSet = CharSet.Auto)]
        public static unsafe extern bool Ordered(Envelope msg, ref Batch Batch1, ref Batch Batch2, int index);

        [DllImport("orderer_kernel.dll", CharSet = CharSet.Auto)]
        public static extern int main();
    }

    class Program
    {
        public const bool useGPU = true; //Change for comparison purposes

        public const int Timeout = 120; //Timeout length in seconds
        public static DateTime Timer = new DateTime();

        public static BlockCutter Cutter = new BlockCutter();
        public static BlockWriter Writer;

        public GPUWrapper GPUCode = new GPUWrapper();

        public static int BlockCount = 0;
        public static long OrderTime = 0;
        public static long OrderTimeCut = 0;
        public static int MsgNum = 0;
        //public static long TotalHandleTime = 0;

        public static unsafe void HandleMessage(Message msg, int index)
        {
            //Stopwatch testsw = new Stopwatch();
            //testsw.Start();


            //Check if configmsg's payload is initialized. If not, it's a normal message.
            if ((IntPtr)msg.ConfigMsg.Payload != IntPtr.Zero) //It's a config message
            {
                Console.WriteLine("[HandleMessage] config message");
                //Cut pending batch so we can make and send the block then a config block
                Batch newbatch = GPUWrapper.Cut(); 
                if (newbatch.MsgCount > 0)
                {
                    Task.Factory.StartNew(() => MakeBlock(newbatch, false));
                }
                //Now make a config block
                Batch newconfigbatch = new Batch();
                newconfigbatch.Messages[0] = msg.ConfigMsg;
                Task.Factory.StartNew(() => MakeBlock(newconfigbatch, true));
            }
            else //It's a normal message
            {
                Console.WriteLine("[HandleMessage] normal message");
                Batch Batch1 = new Batch();
                Batch Batch2 = new Batch();
                //Stopwatch sw1 = new Stopwatch();
                //sw1.Start();
                bool IsPendingBatch = GPUWrapper.Ordered(msg.NormalMsg, ref Batch1, ref Batch2, index);
                //sw1.Stop();
                //if(Batch2.MsgCount > 0)
                //{
                //    OrderTimeCut += sw1.ElapsedMilliseconds;
                //}
                //else
                //{
                //    OrderTime += sw1.ElapsedMilliseconds;
                //}
                //string output = "Time taken: " + sw1.ElapsedMilliseconds + " ms" + Environment.NewLine;
                //Console.WriteLine(output);

                //Ordered returns 0, 1, or 2 batches, process each one returned into a block
                if (Batch1.MsgCount > 0)
                {
                    Task.Factory.StartNew(() => MakeBlock(Batch1, false));
                }
                if (Batch2.MsgCount > 0)
                {
                    Task.Factory.StartNew(() => MakeBlock(Batch2, false)); 
                }

                //Handle setting of timer
                bool timernull = Timer == new DateTime(); //Timer is default value
                if (!timernull && !IsPendingBatch) //There is a timer but no pending batches, unnecessary
                {
                    Timer = new DateTime(); //Set timer to default value
                }
                else if (timernull && IsPendingBatch)
                {
                    Timer = DateTime.Now; //Start timer
                }

            }

            //testsw.Stop();
            //TotalHandleTime += testsw.ElapsedMilliseconds;
            //Console.WriteLine("Time taken: " + testsw.ElapsedMilliseconds);
        }

        public static unsafe void HandleMessageCPU(SafeMessage msg, int index)
        {
            Console.WriteLine(String.Format("HandleMessageCPU {0}", index));
            //Check if configmsg is initialized. If not, it's a normal message.
            bool config = false;

            if (config) //It's a config message
            {
                //Cut pending batch so we can make and send the block then a config block
                SafeBatch newbatch = Cutter.Cut();
                Console.WriteLine("Config Message, newbatch message count={0}", newbatch.Messages.Count());
                if (newbatch.Messages.Count() > 0)
                {
                    Task.Factory.StartNew(() => MakeBlockCPU(newbatch, false));
                }
                //Now make a config block
                SafeBatch newconfigbatch = new SafeBatch();
                newconfigbatch.Messages = new SafeEnvelope[1];
                newconfigbatch.Messages[0] = msg.ConfigMsg;
                Task.Factory.StartNew(() => MakeBlockCPU(newconfigbatch, true));
            }
            else //It's a normal message
            {
                //Stopwatch sw1 = new Stopwatch();
                //sw1.Start();
                SafeOrderedReturn OR = Cutter.Ordered(msg.NormalMsg, index);
                //sw1.Stop();
                //string output = "Time taken: " + sw1.ElapsedTicks + " ticks" + Environment.NewLine;
                //Console.WriteLine(output);
                //Ordered returns 0, 1, or 2 batches, process each one
                Console.WriteLine("Normal Message, batches: {0}, {1}", OR.Batches[0], OR.Batches[1]);
                Parallel.For(0, 2, i =>
                {
                    if (OR.Batches[i] != null)
                    {
                        Task.Factory.StartNew(() => MakeBlockCPU(OR.Batches[i], false));
                    }
                });

                //Handle setting of timer
                bool pending = OR.IsPendingBatch; //There is a pending batch
                bool timernull = Timer == new DateTime(); //Timer is default value
                if (!timernull && !pending) //There is a timer but no pending batches, unnecessary
                {
                    Timer = new DateTime(); //Set timer to default value
                }
                else if (timernull && pending)
                {
                    Timer = DateTime.Now; //Start timer
                }
            }
        }

        //If timer runs out, cut any pending batches and make them into a block
        //Ensures that pending messages will always be sent through after some timeout
        public static void HandleTimer()
        {
            Batch newbatch = GPUWrapper.Cut();
            if(newbatch.MsgCount > 0)
            {
                Task.Factory.StartNew(() => MakeBlock(newbatch, false));
            }
        }

        //Uses blockwriter to create a new block and then write it down
        static void MakeBlock(Batch newbatch, bool config)
        {
            DateTime blockwritetimer = DateTime.Now;
            Block newblock = Writer.CreateNextBlock(newbatch);
            if(config)
            {
                Writer.WriteConfigBlock(newblock);
            }
            else
            {
                Interlocked.Increment(ref BlockCount);
                //Writer.WriteBlock(newblock);
            }

            //string blockoutput = "Creating " + (config ? "config" : "normal") + " block " + newblock.Header.Number + Environment.NewLine;
            //blockoutput += "Message count: " + newbatch.MsgCount + Environment.NewLine;
            //blockoutput += "Time taken: " + (DateTime.Now - blockwritetimer).TotalMilliseconds + "ms" + Environment.NewLine;
            //blockoutput += "---------------------------------------------------------------------" + Environment.NewLine;
            //Console.WriteLine(blockoutput);
        }

        //Uses blockwriter to create a new block and then write it down
        static void MakeBlockCPU(SafeBatch newbatch, bool config)
        {
            DateTime blockwritetimer = DateTime.Now;
            Block newblock = Writer.CreateNextBlockCPU(newbatch);
            if (config)
            {
                Writer.WriteConfigBlock(newblock);
            }
            else
            {
                Interlocked.Increment(ref BlockCount);
                //Writer.WriteBlock(newblock);
            }
            string blockoutput = "Creating " + (config ? "config" : "normal") + " block " + newblock.Header.Number + Environment.NewLine;
            blockoutput += "Message count: " + newbatch.MsgCount + Environment.NewLine;
            blockoutput += "Time taken: " + (DateTime.Now - blockwritetimer).TotalMilliseconds + "ms" + Environment.NewLine;
            blockoutput += "---------------------------------------------------------------------" + Environment.NewLine;
            Console.WriteLine(blockoutput);
        }


        //Waits for a message and makes new thread to handle when received
        static unsafe void Main(string[] args)
        {
            Writer = new BlockWriter(GetPrevHeader());
            GPUWrapper.main(); //Initialize GPU
            //Initialize 100 transactions
            int index = 0;
            int testnum = 10;
            int finalblockcount = testnum / 10;
            //int finalblockcount = testnum;
            List<SafeMessage> msgsjson = JsonConvert.DeserializeObject<List<SafeMessage>>(File.ReadAllText("messages.json"));
            bool printed = false;
            Stopwatch sw = new Stopwatch();
            sw.Start();
            for(int i = 0; i < testnum; i++)
            {
                SafeMessage message = msgsjson.ElementAt(index);
                if (message.ConfigSeq != null) //If message exists
                {
                    if (useGPU)
                    {
                        Console.WriteLine(String.Format("Test {0}, HandleMessage posted", i));
                        Message unsafemessage = ConvertToUnsafe(message);
                        int temp = i; //Concurrency issue happens if I don't do this
                        Task t = Task.Factory.StartNew(() => HandleMessage(unsafemessage, temp)); //Run message handler
                        t.Wait();
                    }
                    else //Use CPU
                    {
                        Console.WriteLine(String.Format("Test {0}, HandleMessageCPU posted", i));
                        int temp = i;
                        Task.Factory.StartNew(() => HandleMessageCPU(message, temp)); //Run message handler
                    }
                }
                else if (Timer != new DateTime() && (DateTime.Now - Timer).TotalSeconds > Timeout)
                {
                    Console.WriteLine(String.Format("Test {0}, HandleTimer posted", i));
                    Timer = new DateTime(); //Reset timer
                    Task.Factory.StartNew(() => HandleTimer());
                }
                index = (index + 1) % 500;
            }

            while(true)
            {
                if (BlockCount >= (finalblockcount) && !printed)
                {
                    sw.Stop();
                    Console.WriteLine("Time to process " + testnum + " transactions: " + sw.ElapsedMilliseconds + " ms");
                    Console.WriteLine("Transactions per second: " + (1000 * (double)testnum / sw.ElapsedMilliseconds) + " tps");
                    //Console.WriteLine("Avg time to order (no batch returned): " + (OrderTime / (double)(testnum - finalblockcount)) + " ms");
                    //Console.WriteLine("Avg time to fully handle msg: " + (TotalHandleTime / (double)testnum) + " ms"); //total handle time is consistently .01ms more than avg order time
                    //Console.WriteLine("Avg time to order (batch returned): " + (OrderTimeCut / (double)finalblockcount) + " ms");
                    printed = true;
                    //Environment.Exit(0);
                }
            }

        }

        static unsafe Message ConvertToUnsafe(SafeMessage msg)
        {
            Message usmsg = new Message();
            usmsg.ConfigSeq = msg.ConfigSeq;

            usmsg.ConfigMsg = new Envelope();

            if(msg.ConfigMsg.Payload.Length > 0)
            {
                IntPtr cp = Marshal.AllocHGlobal(msg.ConfigMsg.Payload.Length);
                Marshal.Copy(msg.ConfigMsg.Payload, 0, cp, msg.ConfigMsg.Payload.Length);
                usmsg.ConfigMsg.Payload = (byte*)cp.ToPointer();
            }

            if (msg.ConfigMsg.Signature.Length > 0)
            {
                IntPtr cs = Marshal.AllocHGlobal(msg.ConfigMsg.Signature.Length);
                Marshal.Copy(msg.ConfigMsg.Signature, 0, cs, msg.ConfigMsg.Signature.Length);
                usmsg.ConfigMsg.Signature = (byte*)cs.ToPointer();
            }

            usmsg.NormalMsg = new Envelope();

            if (msg.NormalMsg.Payload.Length > 0)
            {
                IntPtr np = Marshal.AllocHGlobal(msg.NormalMsg.Payload.Length);
                Marshal.Copy(msg.NormalMsg.Payload, 0, np, msg.NormalMsg.Payload.Length);
                usmsg.NormalMsg.Payload = (byte*)np.ToPointer();
            }

            if (msg.NormalMsg.Signature.Length > 0)
            {
                IntPtr ns = Marshal.AllocHGlobal(msg.NormalMsg.Signature.Length);
                Marshal.Copy(msg.NormalMsg.Signature, 0, ns, msg.NormalMsg.Signature.Length);
                usmsg.NormalMsg.Signature = (byte*)ns.ToPointer();
            }

            return usmsg;
        }

        static BlockHeader GetPrevHeader()
        {
            string prevtext = File.ReadAllText("previousheader.json");
            return JsonConvert.DeserializeObject<BlockHeader>(prevtext);
        }

        static SafeMessage GetMessage()
        {
            try
            {
                string messagestext = File.ReadAllText("messages.json");
                //Get all messages from the json file
                List<SafeMessage> messages = JsonConvert.DeserializeObject<List<SafeMessage>>(messagestext);
                SafeMessage message = messages[0]; //Get first element
                messages.RemoveAt(0); //Remove that element
                string removedtext = JsonConvert.SerializeObject(messages); //Convert list - first object to string
                //May error because file is open, try until successful
                while (true)
                {
                    try
                    {
                        if(messages.Count > 0) //Write remaining messages
                        {
                            File.WriteAllText("messages.json", removedtext);
                            break;
                        }
                        else //Write empty list
                        {
                            File.WriteAllText("messages.json", "[]");
                            break;
                        }
                    }
                    catch { }
                }
                return message;
            }
            catch //No message available
            {
                return new SafeMessage();
            }

        }

        static SafeMessage GetMessageDontRemove(int index)
        {
            try
            {
                string messagestext = File.ReadAllText("messages.json");
                //Get all messages from the json file
                List<SafeMessage> messages = JsonConvert.DeserializeObject<List<SafeMessage>>(messagestext);
                SafeMessage message = messages[index]; //Get first element
                //May error because file is open, try until successful
                return message;
            }
            catch //No message available
            {
                return new SafeMessage();
            }
        }


    }
}
