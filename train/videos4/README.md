# VQ-VAE For Video

use frame difference as input for VQ-VAE. The first frame will be kept, the rest frame is the delta compared to the 
previous frame.

Couldn't learn anything. Error not going down.

input = frame[0] || frames[1:] - frames[:-1]