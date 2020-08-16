using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace OrdererTest
{
    class BlockHeader
    {
        public ulong Number; //Unsigned 64 bit int
        public byte[] PreviousHash;
        public byte[] DataHash;
    }

    class BlockData
    {
        public byte[][] Data;
    }

    class BlockMetadata
    {
        public byte[][] Metadata;
    }

    class Block
    {
        public BlockHeader Header;
        public BlockData Data;
        public BlockMetadata Metadata;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 0)]
    unsafe struct Envelope
    {
        public byte* Payload;
        public byte* Signature;
    }

    class SafeEnvelope
    {
        public byte[] Payload;
        public byte[] Signature;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 0)]
    unsafe struct Batch
    {
        public Envelope* Messages;
        public int MsgCount;
    }

    class SafeBatch
    {
        public SafeEnvelope[] Messages;
        public int MsgCount;
    }


    class Message
    {
        public byte[] ConfigSeq;
        public Envelope ConfigMsg;
        public Envelope NormalMsg;
    }

    class SafeMessage
    {
        public byte[] ConfigSeq;
        public SafeEnvelope ConfigMsg;
        public SafeEnvelope NormalMsg;
    }

    class SafeOrderedReturn
    {
        public SafeBatch[] Batches;
        public bool IsPendingBatch;
    }
}
