__constant__ const uint32_t K_SHA256[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U
};

// Fast byte swap using GPU intrinsic (1 cycle!)
#define BSWAP32(x) (__byte_perm((x), 0, 0x0123))

// Fast rotate using GPU intrinsic (1 cycle!)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 320)
    #define ROTR(x, n) (__funnelshift_r((x), (x), (n)))
#else
    #define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#endif

// SHA-256 operations
#define CH(x, y, z)  ((x & y) ^ (~x & z))
#define MAJ(x, y, z) ((x & y) ^ (x & z) ^ (y & z))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ (x >> 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ (x >> 10))

// ============================================================================
// CORE SHA-256 COMPRESSION (using 16-word rolling buffer)
// ============================================================================

__device__ __forceinline__ void sha256_compress(uint32_t state[8], uint32_t w[16]) {
    // Initialize working variables
    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    // Main compression loop - fully unrolled with rolling buffer
    #pragma unroll
    for (int t = 0; t < 64; t++) {
        uint32_t wt;
        
        if (t < 16) {
            wt = w[t];
        } else {
            // Compute using circular buffer (saves memory!)
            uint32_t s0 = SIG0(w[(t + 1) & 0x0f]);   // w[t-15]
            uint32_t s1 = SIG1(w[(t + 14) & 0x0f]);  // w[t-2]
            wt = w[t & 0x0f] = w[t & 0x0f] + s0 + w[(t + 9) & 0x0f] + s1;
        }

        uint32_t t1 = h + EP1(e) + CH(e, f, g) + K_SHA256[t] + wt;
        uint32_t t2 = EP0(a) + MAJ(a, b, c);
        
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    // Add to state
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

// ============================================================================
// SPECIALIZED: SHA-256 for 33 bytes (compressed pubkey)
// NO divergence when all threads hash 33 bytes!
// ============================================================================

__device__ void sha256_33(const uint8_t *data, uint8_t hash[32]) {
    uint32_t state[8] = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U
    };

    uint32_t w[16];  // 16-word rolling buffer (64 bytes per thread)

    // Load 32 bytes (8 words) and convert to big-endian
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        w[i] = BSWAP32(((uint32_t*)data)[i]);
    }

    // Last byte + padding (0x80) + zeros
    w[8] = ((uint32_t)data[32] << 24) | 0x00800000U;

    #pragma unroll
    for (int i = 9; i < 15; i++) {
        w[i] = 0;
    }

    // Length in bits: 33 * 8 = 264 (big-endian)
    w[15] = 264;

    // Process single block
    sha256_compress(state, w);

    // Output hash (convert to big-endian)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        ((uint32_t*)hash)[i] = BSWAP32(state[i]);
    }
}

// ============================================================================
// SPECIALIZED: SHA-256 for 65 bytes (uncompressed pubkey)
// NO divergence when all threads hash 65 bytes!
// ============================================================================

__device__ void sha256_65(const uint8_t *data, uint8_t hash[32]) {
    uint32_t state[8] = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U
    };

    uint32_t w[16];  // 16-word rolling buffer

    // ========== BLOCK 1: First 64 bytes ==========
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        w[i] = BSWAP32(((uint32_t*)data)[i]);
    }

    sha256_compress(state, w);

    // ========== BLOCK 2: Last byte + padding ==========
    w[0] = ((uint32_t)data[64] << 24) | 0x00800000U;

    #pragma unroll
    for (int i = 1; i < 15; i++) {
        w[i] = 0;
    }

    // Length in bits: 65 * 8 = 520 (big-endian)
    w[15] = 520;

    sha256_compress(state, w);

    // Output
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        ((uint32_t*)hash)[i] = BSWAP32(state[i]);
    }
}

// ============================================================================
// OPTIMIZED RIPEMD-160 FOR BITCOIN ADDRESS GENERATION
// ============================================================================

// Move lookup tables to constant memory (cached, fast broadcast)
__constant__ const uint32_t RIPEMD_K1[5] = {
    0x00000000U, 0x5A827999U, 0x6ED9EBA1U, 0x8F1BBCDCU, 0xA953FD4EU
};

__constant__ const uint32_t RIPEMD_K2[5] = {
    0x50A28BE6U, 0x5C4DD124U, 0x6D703EF3U, 0x7A6D76E9U, 0x00000000U
};

// Rotation indices for left line (5 rounds × 16 steps)
__constant__ const uint8_t RL[80] = {
    // Round 1
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    // Round 2
    7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    // Round 3
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
    // Round 4
    1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
    // Round 5
    4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
};

// Rotation indices for right line (5 rounds × 16 steps)
__constant__ const uint8_t RR[80] = {
    // Round 1
    5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
    // Round 2
    6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    // Round 3
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
    // Round 4
    8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
    // Round 5
    12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
};

// Shift amounts for left line (5 rounds × 16 steps)
__constant__ const uint8_t SL[80] = {
    // Round 1
    11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
    // Round 2
    7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    // Round 3
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
    // Round 4
    11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
    // Round 5
    9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
};

// Shift amounts for right line (5 rounds × 16 steps)
__constant__ const uint8_t SR[80] = {
    // Round 1
    8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
    // Round 2
    9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    // Round 3
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
    // Round 4
    15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
    // Round 5
    8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
};

// Optimized rotate left using CUDA intrinsic (SM 3.2+)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 320)
    #define RIPEMD_ROL(x, n) (__funnelshift_l((x), (x), (n)))
#else
    #define RIPEMD_ROL(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
#endif

// RIPEMD-160 functions (as macros for inlining)
#define RIPEMD_F1(x, y, z) ((x) ^ (y) ^ (z))
#define RIPEMD_F2(x, y, z) (((x) & (y)) | (~(x) & (z)))
#define RIPEMD_F3(x, y, z) (((x) | ~(y)) ^ (z))
#define RIPEMD_F4(x, y, z) (((x) & (z)) | ((y) & ~(z)))
#define RIPEMD_F5(x, y, z) ((x) ^ ((y) | ~(z)))

// ============================================================================
// SPECIALIZED: RIPEMD-160 for 32-byte input (SHA-256 output)
// This is the ONLY size needed for Bitcoin address generation!
// ============================================================================

__device__ void ripemd160(const uint8_t data[32], uint8_t hash[20]) {
    // ========================================================================
    // STEP 1: Initialize state
    // ========================================================================
    uint32_t h0 = 0x67452301U;
    uint32_t h1 = 0xEFCDAB89U;
    uint32_t h2 = 0x98BADCFEU;
    uint32_t h3 = 0x10325476U;
    uint32_t h4 = 0xC3D2E1F0U;

    // ========================================================================
    // STEP 2: Load 32-byte input as 8 words (little-endian)
    // ========================================================================
    uint32_t w[16];

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        w[i] = ((uint32_t)data[i*4 + 0]) |
               ((uint32_t)data[i*4 + 1] << 8) |
               ((uint32_t)data[i*4 + 2] << 16) |
               ((uint32_t)data[i*4 + 3] << 24);
    }

    // ========================================================================
    // STEP 3: Padding for 32-byte input (PRECOMPUTED - NO CONDITIONALS!)
    // ========================================================================
    // 32 bytes = 256 bits
    // Padding: 0x80 followed by zeros, then length
    // Total: 32 + 1 + 23 + 8 = 64 bytes = 512 bits = 1 block
    
    w[8] = 0x00000080U;  // 0x80 padding byte (little-endian)
    w[9] = 0;
    w[10] = 0;
    w[11] = 0;
    w[12] = 0;
    w[13] = 0;
    w[14] = 256;  // Length in bits: 32 * 8 = 256 (little-endian)
    w[15] = 0;    // High word of length (always 0 for 32 bytes)

    // ========================================================================
    // STEP 4: Initialize working variables (two parallel lines)
    // ========================================================================
    uint32_t al = h0, bl = h1, cl = h2, dl = h3, el = h4;  // Left line
    uint32_t ar = h0, br = h1, cr = h2, dr = h3, er = h4;  // Right line

    // ========================================================================
    // STEP 5: LEFT LINE - 5 rounds × 16 steps = 80 steps
    // ========================================================================
    
    // Round 1: F1 function
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        uint32_t t = al + RIPEMD_F1(bl, cl, dl) + w[RL[j]] + RIPEMD_K1[0];
        al = RIPEMD_ROL(t, SL[j]) + el;
        cl = RIPEMD_ROL(cl, 10);
        // Rotate variables: a<-e, b<-a, c<-b, d<-c, e<-d
        uint32_t tmp = el; el = dl; dl = cl; cl = bl; bl = al; al = tmp;
    }

    // Round 2: F2 function
    #pragma unroll
    for (int j = 16; j < 32; j++) {
        uint32_t t = al + RIPEMD_F2(bl, cl, dl) + w[RL[j]] + RIPEMD_K1[1];
        al = RIPEMD_ROL(t, SL[j]) + el;
        cl = RIPEMD_ROL(cl, 10);
        uint32_t tmp = el; el = dl; dl = cl; cl = bl; bl = al; al = tmp;
    }

    // Round 3: F3 function
    #pragma unroll
    for (int j = 32; j < 48; j++) {
        uint32_t t = al + RIPEMD_F3(bl, cl, dl) + w[RL[j]] + RIPEMD_K1[2];
        al = RIPEMD_ROL(t, SL[j]) + el;
        cl = RIPEMD_ROL(cl, 10);
        uint32_t tmp = el; el = dl; dl = cl; cl = bl; bl = al; al = tmp;
    }

    // Round 4: F4 function
    #pragma unroll
    for (int j = 48; j < 64; j++) {
        uint32_t t = al + RIPEMD_F4(bl, cl, dl) + w[RL[j]] + RIPEMD_K1[3];
        al = RIPEMD_ROL(t, SL[j]) + el;
        cl = RIPEMD_ROL(cl, 10);
        uint32_t tmp = el; el = dl; dl = cl; cl = bl; bl = al; al = tmp;
    }

    // Round 5: F5 function
    #pragma unroll
    for (int j = 64; j < 80; j++) {
        uint32_t t = al + RIPEMD_F5(bl, cl, dl) + w[RL[j]] + RIPEMD_K1[4];
        al = RIPEMD_ROL(t, SL[j]) + el;
        cl = RIPEMD_ROL(cl, 10);
        uint32_t tmp = el; el = dl; dl = cl; cl = bl; bl = al; al = tmp;
    }

    // ========================================================================
    // STEP 6: RIGHT LINE - 5 rounds × 16 steps = 80 steps
    // ========================================================================
    
    // Round 1: F5 function
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        uint32_t t = ar + RIPEMD_F5(br, cr, dr) + w[RR[j]] + RIPEMD_K2[0];
        ar = RIPEMD_ROL(t, SR[j]) + er;
        cr = RIPEMD_ROL(cr, 10);
        uint32_t tmp = er; er = dr; dr = cr; cr = br; br = ar; ar = tmp;
    }

    // Round 2: F4 function
    #pragma unroll
    for (int j = 16; j < 32; j++) {
        uint32_t t = ar + RIPEMD_F4(br, cr, dr) + w[RR[j]] + RIPEMD_K2[1];
        ar = RIPEMD_ROL(t, SR[j]) + er;
        cr = RIPEMD_ROL(cr, 10);
        uint32_t tmp = er; er = dr; dr = cr; cr = br; br = ar; ar = tmp;
    }

    // Round 3: F3 function
    #pragma unroll
    for (int j = 32; j < 48; j++) {
        uint32_t t = ar + RIPEMD_F3(br, cr, dr) + w[RR[j]] + RIPEMD_K2[2];
        ar = RIPEMD_ROL(t, SR[j]) + er;
        cr = RIPEMD_ROL(cr, 10);
        uint32_t tmp = er; er = dr; dr = cr; cr = br; br = ar; ar = tmp;
    }

    // Round 4: F2 function
    #pragma unroll
    for (int j = 48; j < 64; j++) {
        uint32_t t = ar + RIPEMD_F2(br, cr, dr) + w[RR[j]] + RIPEMD_K2[3];
        ar = RIPEMD_ROL(t, SR[j]) + er;
        cr = RIPEMD_ROL(cr, 10);
        uint32_t tmp = er; er = dr; dr = cr; cr = br; br = ar; ar = tmp;
    }

    // Round 5: F1 function
    #pragma unroll
    for (int j = 64; j < 80; j++) {
        uint32_t t = ar + RIPEMD_F1(br, cr, dr) + w[RR[j]] + RIPEMD_K2[4];
        ar = RIPEMD_ROL(t, SR[j]) + er;
        cr = RIPEMD_ROL(cr, 10);
        uint32_t tmp = er; er = dr; dr = cr; cr = br; br = ar; ar = tmp;
    }

    // ========================================================================
    // STEP 7: Combine results from both lines
    // ========================================================================
    uint32_t t = h1 + cl + dr;
    h1 = h2 + dl + er;
    h2 = h3 + el + ar;
    h3 = h4 + al + br;
    h4 = h0 + bl + cr;
    h0 = t;

    // ========================================================================
    // STEP 8: Output hash (convert to byte array, little-endian)
    // ========================================================================
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        uint32_t word;
        switch(i) {
            case 0: word = h0; break;
            case 1: word = h1; break;
            case 2: word = h2; break;
            case 3: word = h3; break;
            case 4: word = h4; break;
        }
        
        hash[i*4 + 0] = word & 0xFF;
        hash[i*4 + 1] = (word >> 8) & 0xFF;
        hash[i*4 + 2] = (word >> 16) & 0xFF;
        hash[i*4 + 3] = (word >> 24) & 0xFF;
    }
}
