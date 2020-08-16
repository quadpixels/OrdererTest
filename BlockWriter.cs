using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Security.Cryptography;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;
using Newtonsoft.Json;
using System.Diagnostics;

namespace OrdererTest
{

    class BlockWriter
    {

        public BlockHeader PreviousBlockHeader;

        private SHA256 hasher = SHA256.Create();

        private bool useGPU;

        //Create block out of batch
        unsafe public Block CreateNextBlock(Batch newbatch)
        {
            byte[] previousblockhash = BlockHeaderHash(PreviousBlockHeader);
            BlockData blockdata = new BlockData();
            byte[][] data = new byte[newbatch.MsgCount][];
            //Convert each message to byte format to add to data
            Parallel.For(0, newbatch.MsgCount, i =>
            {
                SafeEnvelope safe = new SafeEnvelope();
                List<byte> pllist = new List<byte>();
                List<byte> siglist = new List<byte>();
                int j = 0;
                while(true)
                {
                    if(newbatch.Messages[i].Payload[j] == 0)
                    {
                        break;
                    }
                    else
                    {

                        pllist.Add(newbatch.Messages[i].Payload[j]);
                    }
                    j++;
                }
                j = 0;
                while (true)
                {
                    if (newbatch.Messages[i].Signature[j] == 0)
                    {
                        break;
                    }
                    else
                    {
                        siglist.Add(newbatch.Messages[i].Signature[j]);
                    }
                    j++;
                }
                safe.Payload = pllist.ToArray();
                safe.Signature = siglist.ToArray();
                data[i] = Marshal(safe);
            });
            blockdata.Data = data;
            Block newblock = NewBlock(PreviousBlockHeader.Number + 1, previousblockhash);
            newblock.Header.DataHash = BlockDataHash(blockdata);
            newblock.Data = blockdata;
            PreviousBlockHeader = newblock.Header;
            return newblock;
        }

        public Block CreateNextBlockCPU(SafeBatch newbatch)
        {
            byte[] previousblockhash = BlockHeaderHash(PreviousBlockHeader);
            BlockData blockdata = new BlockData();
            byte[][] data = new byte[newbatch.Messages.Count()][];
            //Convert each message to byte format to add to data
            Parallel.For(0, newbatch.Messages.Count(), i =>
            {
                data[i] = Marshal(newbatch.Messages[i]);
            });
            blockdata.Data = data;
            Block newblock = NewBlock(PreviousBlockHeader.Number + 1, previousblockhash);
            newblock.Header.DataHash = BlockDataHash(blockdata);
            newblock.Data = blockdata;
            PreviousBlockHeader = newblock.Header;
            return newblock;
        }

        //Saves block as .json, watched for by hyperledger go program
        public void WriteBlock(Block newblock)
        {
            string jsontext = JsonConvert.SerializeObject(newblock);
            File.WriteAllText("block" + newblock.Header.Number + ".json", jsontext);
        }

        public void WriteConfigBlock(Block newblock)
        {
            string jsontext = JsonConvert.SerializeObject(newblock);
            File.WriteAllText("block" + newblock.Header.Number + "-config.json", jsontext);
        }

        // Convert an object to a byte array
        public byte[] Marshal(Object obj)
        {
            string objjson = JsonConvert.SerializeObject(obj);
            byte[] bytes = new byte[objjson.Length];
            GetBytes(objjson.ToCharArray(), objjson.Length, bytes);
            return bytes;
        }

        //Convert each char of string into a byte and add to array
        private void GetBytes(char[] obj, int len, byte[] byteconvert)
        {
            Parallel.For(0, len, i =>
            {
                byteconvert[i] = (byte)obj[i];
            });
        }

        public byte[] BlockHeaderHash(BlockHeader hdr)
        {
            return hasher.ComputeHash(Marshal(hdr)); //Create hash of byte representation of header
        }

        public byte[] BlockDataHash(BlockData data)
        {
            return hasher.ComputeHash(Marshal(data)); //Create hash of byte representation of data
        }

        //Initialize block with supplied values
        public Block NewBlock(ulong SeqNum, byte[] prevhash)
        {
            Block newblock = new Block();
            newblock.Header = new BlockHeader();
            newblock.Header.Number = SeqNum;
            newblock.Header.PreviousHash = prevhash;
            newblock.Metadata = new BlockMetadata();
            newblock.Metadata.Metadata = new byte[1][];
            return newblock;
        }

        public BlockWriter() { } //Constructor for GPU

        //Constructor with initial block header
        public BlockWriter(BlockHeader prev)
        {
            PreviousBlockHeader = prev;
        }
    }
}
