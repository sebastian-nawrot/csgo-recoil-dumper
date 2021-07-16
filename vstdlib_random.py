import numpy as np


NTAB = 32

IA = 16807
IM = 2147483647
IQ = 127773
IR = 2836

NDIV = np.float32(1+(IM-1)/NTAB)
AM = np.float32(1.0/IM)
EPS = np.float32(1.2e-7)
RNMX = np.float32(1.0-EPS)


# Source SDK public\vstdlib\random.h
class UniformRandomStream:
  def __init__(self, seed: np.int32 = np.int32()):
    self.iv = np.empty(NTAB, dtype=np.int32)
    self.set_seed(seed)


  def set_seed(self, seed: np.int32) -> None:
    assert isinstance(seed, np.int32)
    self.idum = seed if seed < 0 else -seed
    self.iy = np.int32()


  def random_float(self, low: np.float32, high: np.float32) -> np.float32:
    assert isinstance(low, np.float32)
    assert isinstance(high, np.float32)
    fl = np.float32(AM * self.generate_random_number())
    if fl > RNMX:
      fl = RNMX
    return (fl * (high - low)) + low


  def generate_random_number(self) -> np.int32:
    if self.idum <= 0 or self.iy == 0:
      self.idum = np.int32(1) if -self.idum < 1 else -self.idum
      for j in reversed(range(NTAB + 7 + 1)):
        k = np.int32(self.idum / IQ)
        self.idum = IA * (self.idum - k * IQ) - IR * k
        
        if self.idum < 0:
          self.idum += IM
        if j < NTAB:
          self.iv[j] = self.idum
      self.iy = self.iv[0]

    k = np.int32(self.idum / IQ)
    self.idum = IA * (self.idum - k * IQ) - IR * k
    if self.idum < 0:
      self.idum += IM
      
    j = np.int32(self.iy / NDIV)
    if j >= NTAB or j < 0:
      j &= NTAB - 1

    self.iy = self.iv[j]
    self.iv[j] = self.idum
    return self.iy