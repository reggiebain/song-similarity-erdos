# Good Composers Borrow, Great Composers Steal
## *Analyzing Audio Similarity using Deep Learning*
#### Authors: Emelie Curl, Tong Shan, Glenn Young, Larsen Linov, Reginald Bain
## Summary
We worked on developing a model using several different approaches to analyze the similarity between songs by directly analyzing the audio. Using audio analysis techniques such as calculating log-mel spectrograms, audio augmentation, and transfer learning we established a way to compare the similarity between audio files for the purpose of detecting potential  plagiarism. Our work focuses on minimizing the so-called *triplet-loss* which aims to maximize the distance between images that are different while simultaenously minimizing the distance between similar images.
## Background
Throughout music history, composers and song writers have borrowed musical elements from each other. Similar chord progressions, rhythms, and melodies can often be found spanning different musical styles, genres, and eras. Our interest in the topic was sparked by many famous examples of composers throughout history borrowing musical structures and motifs from each other. One can hear obvious similarities in the works of Bach and Vivaldi from the baroque era and Mozart/Hayden from the classical era. In the music for Star Wars in 1977, John Williams incorporated similar leitmotifs as composer Igor Stravinsky used in his famous 1913 balet, *The Rite of Spring*. In modern music, courts have adjuticated disputes over similarities between songs. In 2015, for example, artists Robin Thicke and Pharrell Williams were sued for allegedly plagiarizing Marvin Gaye's song "Got to Give it Up" when writing their song "Blurred Lines." Marvin Gaye's estate ultimately won the $7.4 million case. These are just a few examples of audible similarities between musical works. Our project aimed to use deep learning to assess the similarity between music to potentially establish a more robust way to potentially detect music plagiarism.
## Dataset(s)
Our data primarily came from the Million Song Dataset as well as several Kaggle datasets where authors paired known songs with links to preview of the songs from various APIs. We also wrote scripts to find song previews where we did not already have them. We took 30s previews of the songs (in some case taking 10s clips of that 30s due to limitations in both compute and storage), and fed them into Librosa, a Python package built to work with and extract key features from audio. We ultimately focused on the audio from a set of around 50,000 songs that can be found in /data/music_info.csv, a data set containing meta data and preview links to songs from the following Kaggle dataset . This dataset also included information from the LastFM project and a handful of engineered features such as *danceability, liveness, loudness* and many others., a project that aimed to compile songs from the Million Song Dataset and calculate a variety of features ranging from genre tags as well as identify cover songs, and track various musical features of songs. Of primary interest to us was the raw audio of songs rather than meta data or features calculated using said metadata.
## Stakeholders
- Artists looking to ensure their work is not plagiarized by others
- Record companies and courts looking to have an objective measure of song similarity
- Companies looking to better classify and recommend wider arrays of music to listeners
## Key Performance Indicators (KPIs)
- The model minimizes the *triplet-loss* function between a triplet of (anchor, similar, different) audio files to the degree that it will differentite between similar and different audio files.
- The model beats a baseline of calculating the triplet-loss between audio files **without** feeding those audio files into the model. 
## EDA + Feature Engineering
#### Augmenting Audio 
#### Log-Mel Spectrograms
## Modeling
### Transfer Learning
#### ResNet18
#### DistilHuBERT
## Results
## Notable Roadblocks
#### Compute
- Any amateur deep learning project will face compute issues and this was no exception. We made use of the free GPU services offered by both Google Colab and Kaggle, but given the limited time per week the free versions offer, we had to use CPUs in many cases to test model hyperparameters, scrape data, etc. This resulted in extremely long loading times in many cases and limited (given the project timeframe) ability to test every hyperparameter and model architecture to the full extent we desired. For example, fine tuning ResNet18 on 10k triplets of songs for 10 epochs took ~24 hours, even when most model parameters were frozen.
#### Storage
- Storage was a significant limitation as well. Storing the NumPy arrays of log-mel spectrograms of 10k songs in a Pickle file takes over 10GB of storage space. When working locally without significant cloud resources, the data had to be carefully batched so as to not exceed our available hard disk space or overwhelm limited available RAM. This prevented us from trying very large batch sizes (which speeds up training in many cases). 
#### Rate Limiting
- Various APIs impose strict rate limiting often making data scraping time consuming. We used standard techniques where we could to aid with this, but to truly train deep learning architectures one needs far more data than we were able to cobble together.
## References
http://millionsongdataset.com/
https://www.kaggle.com/datasets/undefinenull/million-song-dataset-spotify-lastfm
https://www.last.fm/home