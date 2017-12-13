## Forward Warping Networks - 2 Frame 

- Predict motion for every pixel
- Photometric loss for every pixel
- Input 2 frames
- Output is 2nd frame, the output is also an input of itself
- Color image
- One more layer

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01   |  |  |  | box, m_range=2, batch_size=64, image_size=32, num_frame=2 |
| 02   |  |  |  | box, m_range=2, batch_size=64, image_size=32, num_frame=2, bg_move |
| 03   |  |  |  | mnist, m_range=2, batch_size=64, image_size=32, num_frame=2, bg_move |
| 04   |  |  |  | chair, m_range=2, batch_size=32, image_size=128, num_frame=2 |
| 05   |  |  |  | chair, m_range=2, batch_size=64, image_size=128, num_frame=2 |
| 06   |  |  |  | chair, m_range=2, batch_size=32, image_size=128, num_frame=2, net_depth=13 |
| 07   |  |  |  | chair, m_range=2, batch_size=32, image_size=128, num_frame=2, bg_move |
| 08   |  |  |  | chair, m_range=2, batch_size=32, image_size=128, num_frame=2, net_depth=13, bg_move |

### Take Home Message

