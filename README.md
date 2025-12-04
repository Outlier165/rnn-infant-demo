# rnn-infant-demo

Audio/video PoC for infant state recognition (Colab-ready)

Day1 purpose
- Repository skeleton and Colab notebook to confirm environment and show an example audio waveform and log-mel spectrogram.

Colab notebook (可运行)
- [Paste your Colab share link here ](https://colab.research.google.com/drive/1_doiBr4s6dcImT3qXxO-r1PMAsfxNOZU?usp=sharing)(saved to Drive and set to "Anyone with the link can view").

Quick start
1. Click the Colab link to open the notebook.  
2. Run the first cell to install dependencies.  
3. Run cells in order: imports → demo data download/upload → waveform & log-mel visualization.  
4. To use your own audio, upload via Colab Files or mount Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
