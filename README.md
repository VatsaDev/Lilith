# Lilith
Using the Lilith optimizer on nanogpt, messing with lr, and multiple schdulers

deepseek step based implementation -> [link](https://arxiv.org/html/2401.02954v1#:~:text=rate%20of%20the%20model%20reaches%20its%20maximum%20value%20after%202000%20warmup%20steps%2C%20and%20then%20decreases%20to%2031.6%25%20of%20the%20maximum%20value%20after%20processing%2080%25%20of%20the%20training%20tokens.%20It%20further)


## Running tests

### New lilith versions

 - TODO: Test 21, bs=1024 and bs=2048

 - Test 20, Adam can match lilith at bs=180, testing bs=360 (Yellow and Orange)

![Screen Shot 2024-02-20 at 11 08 44 PM](https://github.com/VatsaDev/Lilith/assets/71975550/d0886d4f-4d73-4004-af2f-a1da99968d4d)
![Screen Shot 2024-02-20 at 11 43 57 PM](https://github.com/VatsaDev/Lilith/assets/71975550/00b10c66-95f8-43a5-a832-cbe9600391da)

 - Test 19, scaling batchsize to 360, appears to be having a similar effect so far, but better, explains euclaise's tests, his bs=1024

![Screen Shot 2024-02-20 at 1 33 30 PM](https://github.com/VatsaDev/Lilith/assets/71975550/1423b591-3b86-49f9-a7c0-b909fcc034aa)


 - Test 18, scaling batchsize to 180 for a try, lr 3e-4, cosine schedule, sota result by a margin, beats adam?! It shows the same behaviour as adamw on large batches, but better? This could be the large scale training optimizer? 

![Screen Shot 2024-02-20 at 12 11 06 PM](https://github.com/VatsaDev/Lilith/assets/71975550/435e1205-4fe6-4adb-9f23-5f7af661bd3c)


 - Test 17, using the deepseek step bases again, first graph 2:4:4, second graph 8:1:1, 8:1:1 is a really successful scheduler, achieved the same val loss as cosine adamw

![Screen Shot 2024-02-20 at 11 18 09 AM](https://github.com/VatsaDev/Lilith/assets/71975550/0bc34e10-4786-4536-b213-324d475e8bca)
![Screen Shot 2024-02-20 at 11 28 17 AM](https://github.com/VatsaDev/Lilith/assets/71975550/799dc67e-fe4c-4977-816a-0334771a98bc)


 - brand new version, due to corruption lost the graphs, but the new good lr is 3e-4, from test 16

 - Test 15, Trying the deepseek based lr steps once again, 2:4:4 (first graph, lr 1e-4, due to numerical instability) and 8:1:1 (second graph, lr 8e-5), the first step change in 2:4:4 worked, but it flatlined afterwards, some progress on that end, while lr on the deepseek values was much much better, almost cosine

![Screen Shot 2024-02-19 at 7 45 18 PM](https://github.com/VatsaDev/Lilith/assets/71975550/79390d23-bb0f-45a1-b3c5-d4feb44a5a98)

![Screen Shot 2024-02-19 at 7 54 41 PM](https://github.com/VatsaDev/Lilith/assets/71975550/64942049-bc5f-49cf-97b5-c7da307dce2e)



 - Test 14, set beta 1 and beta 2 to 0.95 and 0.98, slightly worse, and trial of 0.98 and 0.999999 was even worse but good tuning might give a +1% boost,

![Screen Shot 2024-02-19 at 6 55 46 PM](https://github.com/VatsaDev/Lilith/assets/71975550/60af5711-d679-4792-a8af-9b9312520c36)

![Screen Shot 2024-02-19 at 7 16 49 PM](https://github.com/VatsaDev/Lilith/assets/71975550/ac37589f-6530-46fd-a4c7-3c9b12e63560)


 - Test 13, lr 8e-5, was initially 5e-5, but it was too low, couldnt affect it very well, 8e-5 appears to be an even better initial sweet spot than 1e-4, tho it starts converging

![download (17)](https://github.com/VatsaDev/Lilith/assets/71975550/f1887d4e-4e06-48ba-a263-3fc94a9cfdfc)

 - Test 12, the same as 9, but just testing batch_size and lowering iters for efficiency, slightly above the sota run, but thats expected from larger batches, trains on 1.2x more tokens than before, for 1/3 the time, lilith is scalable, just like AdamW

![download (16)](https://github.com/VatsaDev/Lilith/assets/71975550/da6afddb-761e-4254-8678-3e713d2df1d1)


 - Test 11, changed ema_k from 0 to 1 for better numerical stability, and using cosine lr schedule, lr = 1e-3
 - Note: There is numerical stability, no Nans, but loss is very volatile, literally unlearning

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


