# Lilith
Using the Lilith optimizer on nanogpt, messing with lr, and multiple schdulers

deepseek step based implementation -> [link](https://arxiv.org/html/2401.02954v1#:~:text=rate%20of%20the%20model%20reaches%20its%20maximum%20value%20after%202000%20warmup%20steps%2C%20and%20then%20decreases%20to%2031.6%25%20of%20the%20maximum%20value%20after%20processing%2080%25%20of%20the%20training%20tokens.%20It%20further)


## Running tests

### New lilith versions
 - Test 10, using Triangular lr schedule, literally doesnt want to work, just like the previous tlr spike, gonna stick with multistep or cosine

 ![download (15)](https://github.com/VatsaDev/Lilith/assets/71975550/d1d1f324-a305-4106-80fa-d7a25d587baf)

 - Test 9, the orange bar being the new lilith, lr=1e-4, cosine scheduler, literally matches transformers for awhile, before flattening earlier, but val losses match, at ~1.47, so maybe its just not as prone to overfit?

![download (13)](https://github.com/VatsaDev/Lilith/assets/71975550/5dd47950-cba5-4003-a208-21dd7c17253d)


### Old lilith versions
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

- Test 7, using the deepseek based lr, in yellow, lr 1e-4, 20%,40%,40% partitions, didn't do anything, but that just maybe my infamiliarity with the step based version

![download (11)](https://github.com/VatsaDev/Lilith/assets/71975550/35a5a7e8-213e-49a1-953d-46c00f62cc29)

- Test 8, using the same step partitions in the deepseek paper, teal line, lr 1e-4, 80%,10%,10% partitions, I need to fix it, the lr freaks out and goes to zero, but this optimizer does not seem to like the scheduler whatsoever either, literally no change/drop in all cases

![download (12)](https://github.com/VatsaDev/Lilith/assets/71975550/14a995df-a2ec-4204-a8bd-871d6dc026ed)


