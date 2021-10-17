# VQ-VAE For Video

* stack frames by channel as input
    * `b`: batch size
    * `d`: frame size
  
  ```bash
  (b, d, c, h, w) -> (b, d*c, h, w)
  ```
  
* feed to VQ-VAE

    ```bash
    VqVae1(
      (encoder): Encoder1(
        (blocks): Sequential(
          (input): Sequential(
            (conv_1): Conv2d(90, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
            (relu): ReLU()
            (conv_2): Conv2d(256, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
          )
          (spatial_1): Sequential(
            (block_1): EncoderBlock(
              (id_path): Identity()
              (res_path): Sequential(
                (relu_1): ReLU()
                (conv_1): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (relu_2): ReLU()
                (conv_2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (relu_3): ReLU()
                (conv_3): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (relu_4): ReLU()
                (conv_4): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))
              )
            )
            (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          )
          (spatial_2): Sequential(
            (block_1): EncoderBlock(
              (id_path): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
              (res_path): Sequential(
                (relu_1): ReLU()
                (conv_1): Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (relu_2): ReLU()
                (conv_2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (relu_3): ReLU()
                (conv_3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (relu_4): ReLU()
                (conv_4): Conv2d(32, 128, kernel_size=(1, 1), stride=(1, 1))
              )
            )
            (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          )
          (spatial_3): Sequential(
            (block_1): EncoderBlock(
              (id_path): Conv2d(128, 192, kernel_size=(1, 1), stride=(1, 1))
              (res_path): Sequential(
                (relu_1): ReLU()
                (conv_1): Conv2d(128, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (relu_2): ReLU()
                (conv_2): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (relu_3): ReLU()
                (conv_3): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (relu_4): ReLU()
                (conv_4): Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
              )
            )
            (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
          )
          (spatial_4): Sequential(
            (block_1): EncoderBlock(
              (id_path): Conv2d(192, 256, kernel_size=(1, 1), stride=(1, 1))
              (res_path): Sequential(
                (relu_1): ReLU()
                (conv_1): Conv2d(192, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (relu_2): ReLU()
                (conv_2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (relu_3): ReLU()
                (conv_3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (relu_4): ReLU()
                (conv_4): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
              )
            )
          )
          (output): Sequential(
            (relu): ReLU()
            (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
          )
        )
      )
      (vq_vae): VectorQuantizerEMA(
        (embedding): Embedding(8192, 512)
      )
      (decoder): Decoder1(
        (blocks): Sequential(
          (spatial_1): Sequential(
            (block_1): DecoderBlock(
              (id_path): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
              (res_path): Sequential(
                (relu_1): ReLU()
                (conv_1): Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1))
                (relu_2): ReLU()
                (conv_2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (relu_3): ReLU()
                (conv_3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (relu_4): ReLU()
                (conv_4): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
            )
            (upsample): Upsample(scale_factor=2.0, mode=nearest)
          )
          (spatial_2): Sequential(
            (block_1): DecoderBlock(
              (id_path): Conv2d(256, 192, kernel_size=(1, 1), stride=(1, 1))
              (res_path): Sequential(
                (relu_1): ReLU()
                (conv_1): Conv2d(256, 48, kernel_size=(1, 1), stride=(1, 1))
                (relu_2): ReLU()
                (conv_2): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (relu_3): ReLU()
                (conv_3): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (relu_4): ReLU()
                (conv_4): Conv2d(48, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
            )
            (upsample): Upsample(scale_factor=2.0, mode=nearest)
          )
          (spatial_3): Sequential(
            (block_1): DecoderBlock(
              (id_path): Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1))
              (res_path): Sequential(
                (relu_1): ReLU()
                (conv_1): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1))
                (relu_2): ReLU()
                (conv_2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (relu_3): ReLU()
                (conv_3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (relu_4): ReLU()
                (conv_4): Conv2d(32, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
            )
            (upsample): Upsample(scale_factor=2.0, mode=nearest)
          )
          (spatial_4): Sequential(
            (block_1): DecoderBlock(
              (id_path): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
              (res_path): Sequential(
                (relu_1): ReLU()
                (conv_1): Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))
                (relu_2): ReLU()
                (conv_2): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (relu_3): ReLU()
                (conv_3): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                (relu_4): ReLU()
                (conv_4): Conv2d(16, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
              )
            )
          )
          (output): Sequential(
            (relu_1): ReLU()
            (conv_1): Conv2d(64, 256, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
            (relu_2): ReLU()
            (conv_2): Conv2d(256, 90, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3))
          )
        )
      )
    )
    size 13513242
    ```

