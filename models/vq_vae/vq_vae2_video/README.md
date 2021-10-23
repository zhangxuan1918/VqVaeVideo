# VQ-VAE For Video

## Encoder

* first compress spatially
  * VQ-VAE encoder compresses spatially
  * `b`: batch size
  * `32`: frame size
  
  ```bash
    (b, 32, 3, 256, 256) -> (b, 32, 512, 32, 32)
  ```

* then compress temporally 
  * Another VQ-VAE encoder compresses temporally
  * `(b, 32, 512, 32, 32) -> rearrange`
  * `(b, 512, 32, 32, 32)`, next we do conv3d on the last 3 dims, (32, 32, 32) = (temporal dim, spatial dim1, spatial dim2)
    * filter size: (5, 3, 3)
    *  filter size: (7, 3, 3)
  * `(b, 512, 32, 32, 32) -> conv3d`
  * `(b, 1024, 4, 32, 32) -> rearrange`
  * `(b, 4096, 32, 32) -> quantize -> code to index`
    * vocab dim: `4096`
  * `(b, 16, 16)`
  ```
  
the issue here is that we cannot downsample at spatial dims. But we need to downsample at temporal dim. The
implementation in Torch doesn't support downsample at a chosen dim.

To simplify things, we can do following

* `(b, 32, 512, 32, 32) -> rearrange`
* `(b, 32 * 512, 32, 32) -> conv2d`, problem: the filter is gigantic and cannot be memory efficient
  * can use group to help
* `(b, 4 * 512, 32, 32)`

  
## Decoder

* decode temporally 
  * Another VQ-VAE decoder decodes temporally
  * `(b, 16, 16) -> index to code`
  * `(b, 4096, 16, 16) -> rearrange`
  * `(b, 1024, 4, 16, 16) -> dconv3d`
  * `(b, 512, 32, 32, 32) -> rearrange`
  * `(b, 32, 512, 32, 32)`
  
* decode spatially
  * VQ-VAE encoder compresses spatially
  * `b`: batch size
  * `32`: frame size
  
  ```bash
  (b, 32, 512, 32, 32) -> (b, 32, 3, 256, 256)
  ```

Same can be done in decoder. To simplify things, we can do following

* `(b, 4 * 512, 32, 32)`
* `(b, 32 * 512, 32, 32) -> transposed_conv2d`
* `(b, 512, 32, 32, 32) -> rearrange`