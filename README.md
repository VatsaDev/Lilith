# Lilith
Using the Lilith optimizer on nanogpt, also adding a TLR scheduler to it

## Running tests
 - Test 1, Lilith default params, using cosine LR, AdamW params from Karpathy, cosine LR

![download (1)](https://github.com/VatsaDev/Lilith/assets/71975550/42033ba7-e5a5-4e41-a7a2-e6c0a3e0514f)

 - Test 2, Lilith some slight LR changes(lr 1e-2), using TLR, AdamW params from Karpathy, cosine LR

![download (2)](https://github.com/VatsaDev/Lilith/assets/71975550/b6102282-a299-41f9-97f5-e0fedafd0e0f)

 - Test 3, Lilith lR (3e-4), using cosine lr, adamw the same

![download (6)](https://github.com/VatsaDev/Lilith/assets/71975550/96f0942b-7118-40c6-9ca5-3a08ceab4f24)

- Test 4, current lilith in blue, lr (1e-4), cosine lr
  
![download (7)](https://github.com/VatsaDev/Lilith/assets/71975550/13da4412-2ec9-43e0-83ab-f93a26fa9816)

- Test 5, current lilith in green, lr (5e-5), cosine lr, too low, and the model cant seem to get as low as adamw
- further tests to try and reintroduce TLR, then try a deepseek style stepwise lr

![download (9)](https://github.com/VatsaDev/Lilith/assets/71975550/792014f8-f327-47af-84be-a57b22ed3b1b)

- Test 6, TRL reintroduction(pink), vs sota lilith (blue), and adamW (red), lr 1e-4, didn't go well, TRL is too unstable, will try deepseek stepbased lr later

![download (10)](https://github.com/VatsaDev/Lilith/assets/71975550/657ef261-6175-4abb-a89f-99012e2ee09d)

