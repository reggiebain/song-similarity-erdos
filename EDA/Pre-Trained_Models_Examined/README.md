
In the notebook entitled "SaveMelSpectrogramsasImages.ipynb," an algorithmic process for turning raw audio in the form of wav files into mel spectrograms and then saving those images as .jpg files is implemented. The songs that were selected to undergo this process include 10 pairs of well-know and dissimilar songs:

1. J.S. Bach's Toccata and Fugue in D minor BWV 565 vs. Master Of Puppets by Metallica
2. Michael Jackson's Billie Jean vs. Alice Cooper's Schools Out
3. Yakety Sax by Boots Randolph vs. Vanessa Carlton's A Thousand miles
4. AC⧸DC's Back In Black vs. 4'33'' by John Cage
5. Tequila by The Champs  vs. Frédéric Chopin's Nocturne in E Flat Major (Op. 9 No. 2)
6. Aretha Franklin's Respect vs. Love Theme From Romeo & Juliet by Pyotr Ilyich Tchaikovsky
7. Nirvana's Smells Like Teen Spirit vs. Ave Maria by J.S. Bach and Charles Gounod
8. David Bowie's Heroes vs. OutKast's Hey Ya
9. The Dave Brubeck Quartet's Take Five vs. Public Enemy's Fight the Power
10. Edvard Grieg's Peer Gynt Suite No. 1, Op. 46 I. Morning Mood vs. Pour Some Sugar On Me by Def Leppard

and 20 pairs of songs that were famously involved in court cases surrounding accusations of plagarism: 

1. Vanilla Ice's Ice Ice Baby vs. Under Pressure by Queen and David Bowie
2. the Verve's Bitter Sweet Symphony vs. the Rolling Stone's The Last Time
3. Robin Thicke featuring rapper T.I. and singer Pharrell Williams's Blurred Lines vs. Marvin Gaye's Gotta Give It Up
4. Killing Joke‘s Eighties vs. Nirvana's Come as You are
5. Chuck Berry's You Can't Catch Me vs. The Beatles Come Together
6. Radiohead's Creep vs. Lana Del Rey's Get Free
7. Radiohead's Creep vs. The Air That I Breathe by The Hollies
8. George Harrison's My Sweet Lord vs. the Chiffons’ He’s So Fine
9. Katy Perry featuring Juicy J.'s Dark Horse vs. Flame's Joyful Noise
10. Rod Stewart's Do You Think I'm Sexy vs. Jorge Ben jor's Taj Mahal
11. Ray Parker Jr.'s Ghostbusters vs. Huey Lewis's I Want a New Drug
12. Olivia Rodrigo's Good 4 U vs. Paramore's Misery Business
13. Oasis's Shakermaker vs. The New Seekers's I’d Like To Teach The World To Sing
14. Marvin Gaye's Let's Get It On vs. Ed Sheeran's Thinking Out Loud
15. Led Zeppelin's Stairway to Heaven vs. Spirit's Taurus
16. Sam Smith's Stay with Me vs. Tom Petty's I won't Back Down
17. Beach Boy's Surfin USA vs. Chuck Berry's Sweet Little Sixteen
18. De La Soul's Transmitting Live From Mars vs. The Turtles's You Showed Me
19. Coldplay's Viva La Vida vs. Joe Satriani's If I Could Fly
20. Led Zeppelin's Whole Lotta Love vs. Muddy Waters's You Need Love

In the four notebooks "ResNet18Implementation.ipynb," "ResNet50Implementation.ipynb," "VGG16Implementation.ipynb," and "VGG19Implementation.ipynb," we import four previously created computer vision models ResNet18, ResNet50, VGG16, and VGG19. The goal was to use these models as-is without fine-tuning them to see if after running the mel spectrograms of the previous pairs of mentioned songs through the models, the arrays representing the spectrograms would be reasonably similiar/dissimilar from one another. 

To this end, we also define three algorithmic functions to compute a similiarity score between given vectors. One function is based on scipy.spatial.distance's cosine similarity method, the next is based on sktime.distances's euclidean distance method, and finally the last function is based on dtaidistance's methods to compute the optimal dynamic time warping path between two given time series. Unfortunately, among the chosen models and similarity functions none return terribly accurate similarity scores.

For implementation of VGG16, the similarity scores as given by the dynamic time warping similarity function are all extremely high (close to 1) indicating that every song is similar to every other song (which the human ear will tell you is definitely not the case). The similarity scores as given by the euclidean distance-based similarity function are all extremely low (close to 0) indicating that every song is completely dissimilar to every other song considered (which again is not the case). Finally, the similarity scores as produced by the cosine similarity method are a little more accurate with some of the pairs of dissimilar songs earning relatively low (closer to 0) similarity scores (for example, AC⧸DC's Back In Black vs. 4'33'' by John Cage, Tequila by The Champs vs. Frédéric Chopin's Nocturne in E Flat Major (Op. 9 No. 2), Edvard Grieg's Peer Gynt Suite No. 1, Op. 46 I. Morning Mood vs. Pour Some Sugar On Me by Def Leppard). All other songs, though, were given relatively high similarity scores (at least as large as approximately 0.52). For implementation of VGG19, we obtained very similar results. For ResNet18 and ResNet50 the cosine and DTW similarity are extremely high for all song pairs while the euclidean similiarity is close to 0.50 for each pair. 

In conclusion, training/fine-tuning is obviously very important and completely necessary. Cosine or Euclidean Similarity seem to be better choices for computing a similarity score.  