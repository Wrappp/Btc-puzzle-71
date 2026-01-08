**BTC Puzzle 71 Solver**

A high-performance Bitcoin Puzzle #71 solver written in Python and CUDA, designed to utilize multiple GPUs and operate in a server–client distributed architecture.

**Overview**
This project searches for the private key corresponding to Bitcoin Puzzle 71 by scanning a defined keyspace using GPU acceleration.
It supports running multiple GPU workers on the same machine and distributing work across multiple clients connected to a central server.

**Features:**
Python + CUDA implementation for maximum performance
Multi-GPU support on a single system
Server–client architecture for distributed searching
Scalable keyspace ranges (easy to adapt to any range)
Automatic progress tracking and result persistence
Optional email notification when a solution is found

**Search Range**
Default search range:
0x400000000000000000
to
0x7FFFFFFFFFFFFFFFFF

This range can be easily adjusted to target other puzzles or custom keyspaces.

**Work Distribution**
The total range is split into chunks of:
79,257,600,000 keys per chunk
Each client receives randomly assigned chunks from the server:
Total 14,895,626,675 chunks per client
After completing a chunk, the client reports back to the server with:
chunk_id
completion date & time
client_id

**Match Handling**
When a private key matching Puzzle 71 is found:
The server immediately stops all active clients
The result is saved to disk
An email notification can be sent automatically

**Use Case**
This project is intended for:
Research and experimentation with GPU-accelerated cryptographic searches
If you are interested email me
