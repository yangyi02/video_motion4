## Spatial Transformer Networks - 3 Frame 

- Uses inverse warping
- Predict motion for every pixel
- Photometric loss for every pixel
- Input 2 frames
- Output is 2nd frame, but use 3rd frame to grab pixels to warp
- Color image
- One more layer

### Results

- Column 2 and 3: Improve reconstruction loss over baseline (%) 

| Exp  | Test | Gt   | EPE  | Note |
| ---- | ---- | ---- | ---- | ---- | 
| 01   |  |  |  | box, m_range=2, batch_size=64, image_size=32, num_frame=3 |
| 02   |  |  |  | box, m_range=2, batch_size=64, image_size=32, num_frame=3, bg_move |
| 03   |  |  |  | mnist, m_range=2, batch_size=64, image_size=32, num_frame=3, bg_move |
| 04   |  |  |  | chair, m_range=2, batch_size=32, image_size=128, num_frame=3 |
| 05   |  |  |  | chair, m_range=2, batch_size=64, image_size=128, num_frame=3 |
| 06   |  |  |  | chair, m_range=2, batch_size=32, image_size=128, num_frame=3, net_depth=13 |
| 07   |  |  |  | chair, m_range=2, batch_size=32, image_size=128, num_frame=3, bg_move |
| 08   |  |  |  | chair, m_range=2, batch_size=32, image_size=128, num_frame=3, net_depth=13, bg_move |

### Take Home Message

