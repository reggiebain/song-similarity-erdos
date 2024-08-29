# %%
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import librosa
# %% Show Melspectrogram
def show_melspect(audio_clip, sr):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_clip, sr=sr)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    time_vec = np.arange(mel_spectrogram_db.shape[-1]) * 512 / sr
    mel_frequencies = librosa.mel_frequencies(n_mels=mel_spectrogram.shape[0], fmin=0, fmax=8000)
    fig = px.imshow(
        mel_spectrogram_db,
        x=time_vec,
        y=mel_frequencies,
        # labels={'x': 'Time (s)', 'y': 'Mel Frequency (Hz)', 'color': 'Amplitude (dB)'},
        origin='lower',
        aspect='auto',
        color_continuous_scale='Inferno'
        )
    fig.update_layout(coloraxis_showscale=False,
                      width=400,
                      height=300)
    return fig
# %% Compute Cosine Distance
def compute_cosine_similarity(A, B):
    # Compute cosine similarity
    cosine = np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
    return cosine # Convert to a Python float
# %% Page setup
st.set_page_config(layout="centered", page_icon="🎶", page_title="Song-Similarity")
st.sidebar.header('Song Similarity')
st.markdown(
    '*"Good composers borrow, great ones steal."*'
)
st.title("🎶 Song Similarity")

# %% Load Song Dataframe
@st.cache_data
def read_df(file):
    df = pd.read_pickle(file)
    return df

song_show_df = read_df('song_show_df.pkl')
song_show_df['song_ls'] = song_show_df[['name','artist']].agg(' by '.join, axis=1)
song_show_df['song_ls'] = song_show_df[['song_ls','type']].agg(' - '.join, axis=1)

label_ls = song_show_df['song_ls'].tolist()

# %% Song selection module
options = st.multiselect(
    "⬇️**Select two songs and find out their similarity!**",
    label_ls,
    max_selections=2,
    placeholder='Music list',
    label_visibility="visible"
)

# %% Data viz module
if 'preview0' not in st.session_state:
    st.session_state.preview0 = False
def preview0():
    st.session_state.preview0 = True

if 'preview1' not in st.session_state:
    st.session_state.preview1 = False
def preview1():
    st.session_state.preview1 = True

if 'melspec0' not in st.session_state:
    st.session_state.melspec0 = False
def melspec0():
    st.session_state.melspec0 = True

if 'melspec1' not in st.session_state:
    st.session_state.melspec1 = False
def melspec1():
    st.session_state.melspec1 = True

col1, col2 = st.columns((2))
if len(options) >= 1:
    ind0 = label_ls.index(options[0])
    with col1:
        st.write(options[0])
        st.button("▶️ Preview", key='pv0', on_click=preview0)
        if st.session_state.preview0:
            st.audio(np.array(song_show_df.iloc[ind0]['audio_clip']),sample_rate=22050)
        st.button("Show Melspectrogram", key='ms0', on_click=melspec0)
        if st.session_state.melspec0:
            fig0 = show_melspect(np.array(song_show_df.iloc[ind0]['audio_clip']), 22050)
            st.plotly_chart(fig0)

    if len(options) == 2:
        with col2:
            ind1 = label_ls.index(options[1])
            st.write(options[1])
            st.button("▶️ Preview", key='pv1', on_click=preview1)
            if st.session_state.preview1:
                st.audio(np.array(song_show_df.iloc[ind1]['audio_clip']),sample_rate=22050)
            st.button("Show Melspectrogram", key='ms1', on_click=melspec1)
            if st.session_state.melspec1:
                fig1 = show_melspect(np.array(song_show_df.iloc[ind1]['audio_clip']), 22050)
                st.plotly_chart(fig1)
st.divider()
# %% Song similarity Score module
sim_score = st.button("💡 Compute Song Similarity Score")
if (len(options) == 2) and (sim_score):
    score = compute_cosine_similarity(np.array(song_show_df.iloc[ind0]['embedding']).squeeze(),
                                      np.array(song_show_df.iloc[ind1]['embedding']).squeeze())
    st.header("Cosine Similarity = "+ str(score))

if (len(options) < 2) and (sim_score):
    st.warning('You need to select two songs', icon="⚠️")

st.markdown('''
            **Similarity values** 
            
            A higher value means greater similarity, with a maximum of 1.
            ''')


st.link_button('Try with your own music!', 'https://song-similarity-webapp.streamlit.app/Try_with_your_music')

st.divider()
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Octicons-mark-github.svg/600px-Octicons-mark-github.svg.png",width=30)
st.markdown('''
            [Project github](https://github.com/reggiebain/song-similarity-erdos/tree/main)
            ''')
st.markdown('Made for the Erd&#337;s Institute\'s Deep Learning Boot Camp, Summer 2024.')


