# How the safe's neural network works

Vault 687 opens when you show it a 3-digit code with your fingers. A camera
watches your hand, and a neural network decides how many fingers you're
holding up: 0, 1, 2, 3, 4, 5, or "no hand". No hand-tracking library is
used. The network learned to count fingers only from pictures we took.

## 1. A picture is just numbers

The camera's picture (1920 × 1080 pixels) is shrunk to **128 × 72 pixels**.
Each pixel is three numbers from 0 to 255: how much red, green and blue it has. So the network
starts with 128 × 72 × 3 = **27,648 numbers**. It never "sees" a hand. It
only does math on those numbers.

## 2. Kernels

The network's basic tool is a **kernel**, a pattern finder: a tiny 3 × 3
grid of numbers. It slides across the picture one spot at a time. At each spot it
multiplies the 3 × 3 pixels under it by its own numbers and adds them up.
A big result means "my pattern is here", and a small or negative result
means "not here". Negative results are set to zero.

The output is a new picture, a **pattern map**, that is bright wherever the
pattern was found. For example, one of the network's first kernels
learned to light up **edges**, like the outline of each finger.

## 3. Layers: smaller pictures, more patterns

The network stacks four layers of kernels. After each layer, the
maps are shrunk to half their size by keeping only the strongest value in
every 2 × 2 square. Each new layer looks at the maps the previous layer made,
so it can find bigger, more complicated patterns:

| Layer | Pattern maps | Size of each map | What they tend to find |
|---|---|---|---|
| 1 | 16 | 64 × 36 | edges |
| 2 | 32 | 32 × 18 | curves, corners |
| 3 | 64 | 16 × 9 | parts of fingers |
| 4 | 128 | 8 × 4 | the whole hand |

Nobody told the network to look for edges or fingers. It invented these
patterns itself while learning.

## 4. The vote

Each of the last 128 maps is averaged down to a single number, giving
**128 numbers** that describe the picture. Each of the 7 possible answers
gets a score: every one of those 128 numbers is multiplied by a weight that
belongs to that answer, and the results are added up. The scores are then
turned into chances that add up to 100%. The answer with the highest chance
wins. For the example picture in the video, the network said "3" with 98%
certainty.

## 5. How it learned

All together the network has **294,903 numbers** (the values inside every
kernel, plus the voting weights). They started out random. Learning
means adjusting them:

1. **Guess:** show the network a picture and get its answer.
2. **Check:** compare the answer with the right one, and measure how wrong it was.
3. **Tweak:** change every one of the 294,903 weights a tiny bit in the
   direction that would have made the answer less wrong. (The math for
   working out those directions is called *backpropagation*.)
4. **Repeat** thousands of times.

We collected **8,320 labeled pictures from 5 people**. The network studied
all of them **25 times**, which took about 3 minutes on a laptop. Its
practice score went from 43% after the first round to 99.8% at the end.

Three tricks stopped it from just memorizing the pictures:

- **Shaken-up copies:** every time a picture was used, it was randomly
  mirrored, tilted, zoomed, moved, and made brighter or darker. The network
  never saw the exact same picture twice.
- **Dropout:** while learning, some of the 128 final numbers were randomly
  ignored, so the network couldn't rely on any single one.
- **Fair turns:** people who recorded fewer pictures were picked more often,
  so every person's hand counted equally.

## 6. The quiz: a hand it never saw

To test it fairly, we trained the network *without* one person's pictures
and then quizzed it on that person. A high practice score doesn't prove
much, because it could have memorized those pictures. A good score on a
stranger's hand shows it really learned.

- Learning from 2 people, it scored **80%** on Kabir's hand. It had learned
  a shortcut ("wide-open hand = 5") and called all of his 4s a 5.
- Learning from 4 people, it scored **98.5%**. More variety broke the
  shortcut, so it actually had to count.

**More people beat a fancier network.**

## 7. In the safe

- A digit counts only after the network is at least **80% sure** and gives
  the **same answer for 2 seconds**. Quick wrong guesses are ignored.
- Between digits, you take your hand out of view, so a code like 1-1-3
  doesn't count the same "1" twice.
- A wrong digit resets everything, so a guesser never learns which of the
  other digits were right.

## 8. Running on the Arduino UNO Q

The network is trained with PyTorch on a laptop. To run on the UNO Q, its
294,903 numbers are saved to a file, and the board does the same math with
NumPy. That's the same network, just a different calculator. Both versions
gave the same answer on all 8,480 test pictures. The board makes a guess in
about **71 ms**, roughly 12 guesses a second. It reads the camera, runs the
network, applies the game rules, and controls the LEDs and the servo lock.

---

# Narration script for the video

The video (`how_it_works.mp4`) is 2:44 long and has no sound. Each block
below starts at the time shown. Speak at a relaxed pace; the lines fit their
scenes with a little room to spare.

**0:00 – Title**
> How does our safe know how many fingers you're holding up? It uses a
> neural network. Let's see how, step by step.

**0:09 – The camera takes a picture**
> It starts with the camera. Every moment, it takes a picture of your hand
> over the board.

**0:17 – Shrink it to 128 × 72 pixels**
> First, we shrink the picture to just 128 by 72 pixels. Zoom in, and you can
> see it's really made of tiny colored squares.

**0:27 – Every pixel is just 3 numbers**
> And each pixel is just three numbers: how much red, green and blue it
> has, from 0 to 255. To the computer, your hand is nothing but
> twenty-seven thousand numbers.

**0:40 – A kernel slides over the picture**
> Now the network uses a kernel: a tiny three-by-three grid of
> numbers. It slides across the picture, and everywhere its pattern
> matches, it lights up. This one learned, all by itself, to find edges,
> like the outline of each finger.

**0:57 – 16 kernels, 16 new pictures**
> The first layer has sixteen of these kernels, so one picture becomes
> sixteen pattern maps.

**1:04 – Layers**
> Then it does it again, and again. Four layers in total. Each time, the
> maps get smaller, but there are more of them. Edges become curves, curves
> become parts of fingers, and by the last layer, it's looking at the whole
> hand.

**1:21 – Every answer gets a vote**
> Those last maps are squeezed into 128 numbers. Every possible answer, zero
> through five or no hand, adds them up its own way to get a score. The
> highest score wins. Here, it's 98 percent sure: three fingers.

**1:38 – How it learned: lots of examples**
> But how did it learn all of that? From examples: 8,320 pictures from five
> people. And every time it saw one, we shook it up: mirrored, tilted,
> zoomed, darker or brighter, so it couldn't just memorize.

**1:53 – Guess, check, tweak, repeat**
> Learning is a loop. Guess. Check the answer. Then tweak every one of its
> almost 300,000 weights a tiny bit, so next time it's a little less wrong.
> After 25 rounds, about three minutes, it got almost every practice
> picture right.

**2:12 – The quiz: a hand it never saw**
> The real test is a hand it has never seen. Learning from only two people,
> it scored 80 percent on Kabir's hand. Learning from four people, 98.5
> percent. More people made it smarter.

**2:26 – Opening the safe**
> In the safe, you have to hold each number still for two seconds before it
> counts. Get all three right and Vault 687 opens.

**2:40 – The end**
> Thanks for watching!
