import jittor as jt
import models.dat_jittor 

model = models.dat_jittor.dat_tiny()
inputs = jt.randn(1, 3, 224, 224)
outputs = model(inputs)
x, _, _ = outputs
