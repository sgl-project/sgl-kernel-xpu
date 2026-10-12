// Reuse the shared FMHA dispatch on BMG. The implementation lives in the
// CRI translation unit because Inkling relative attention is a CRI kernel.
#include "flash_attentionXe35.cpp"
