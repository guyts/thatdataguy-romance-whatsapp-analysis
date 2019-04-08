# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 21:18:52 2019

@author: guyts
"""

# 1year anniversary project! (data almost from an entire year of relationship)

import pandas as pd
import numpy as np
import seaborn as sns
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords

## UPDATE THE LINE BELOW WITH YOUR PLOTLY CREDENTIALS
plotly.tools.set_credentials_file(username='', api_key='')

# Reading in the file
df = pd.read_csv('whatsapp_data_may_march_clean.csv',parse_dates=[['date', 'time']])

# removing media messages with not text content
df_no_media = df[df.content != '<Media omitted>']

# aux function to lable days of week
def label_daysofweek (row):
    if row['weekday'] == 0 :
        return 'Monday'
    if row['weekday'] == 1 :
        return 'Tuesday'
    if row['weekday'] == 2 :
        return 'Wednesday'
    if row['weekday'] == 3 :
        return 'Thursday'
    if row['weekday'] == 4 :
        return 'Friday'
    if row['weekday'] == 5 :
        return 'Saturday'
    if row['weekday'] == 6 :
        return 'Sunday'
    return


# adding weekday column:
df['weekday'] = df['date_time'].apply(lambda x: x.weekday()) # list of the days of the week

# Brainstorming:
'''
- who messaged more in general?
- how does travel (together, separate) affects communication?
- what times of days do we usually chat in? 
- do we chat more on weekdays/weekends?
- who sends the first message in the monring more often?
- who initiates conversation in what percentage of the time, in general?
- whose response time is longer?

Content:
- exclude stopwords english
- which words does each of us use more often?
- which nicknames stand out?
- who is more positive and who is more negative? (lookup online)
- was any word used more often/less often with time?
- who sent more media?
- how many time did each say the word love, and when did it start?
'''

### NUMBERS

## overall count of messages:
guy_tot = df[df.sender == 'Guy'].sender.count()
oli_tot = df[df.sender == 'Oliver'].sender.count()

df_timeline = df.groupby([df['date_time'].dt.date, 'sender']).content.count().reset_index()

guy_std = np.std(df_timeline[df_timeline['sender']=='Guy'].content)
oli_std = np.std(df_timeline[df_timeline['sender']=='Oliver'].content)
## timeline of message sending by both:
traceG = go.Scatter(
        x = list(df_timeline[df_timeline['sender']=='Guy'].date_time),
        y = list(df_timeline[df_timeline['sender']=='Guy'].content),
        mode = 'lines',
        name = 'by GT',
        marker = dict(
                color = 'rgb(221,46,107)'
                )
        )

traceO = go.Scatter(
        x = list(df_timeline[df_timeline['sender']=='Oliver'].date_time),
        y = list(df_timeline[df_timeline['sender']=='Oliver'].content),
        mode = 'lines',
        name = 'by OJ',
        marker = dict(
                color = 'rgb(0,102,153)'
                )
        )

data = [traceO, traceG]
# slider set up:
layout = dict(
    title='All Messages Ever Sent, Timeline',
    legend = dict(orientation="h"),
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(count=1,
                    label='YTD',
                    step='year',
                    stepmode='todate'),
                dict(count=1,
                    label='1y',
                    step='year',
                    stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    )
)


fig = dict(data=data, layout=layout)
py.iplot(fig, filename='analysis2-messaging-trends')


# grouping by sender and by date to sum daily count:
df_daily_counts = df.groupby([df['date_time'].dt.date, 'sender']).content.count().reset_index()

# plotting:
# did it above

## weekday averages from each sender:
df_weekly = df.groupby(['sender',df['date_time'].dt.day,'weekday']).content.count().reset_index()
df_weekly['weekday_w'] = df_weekly.apply(lambda row: label_daysofweek (row),axis=1)
df_weekly_avg = df_weekly.groupby(['weekday','sender']).content.mean().reset_index()

df_weekly = df_weekly.sort_values(by='weekday')
df_weekly = df_weekly[df_weekly['content']<700]
# plotting:
x = df_weekly[df_weekly['sender']=='Guy'].weekday_w.tolist()
y_g= df_weekly[df_weekly['sender']=='Guy'].content.tolist()
y_o= df_weekly[df_weekly['sender']=='Oliver'].content.tolist()

trace_g = go.Box(
    y=y_g,
    x=df_weekly[df_weekly['sender']=='Guy'].weekday_w.tolist(),
    name='by GT',
    marker = dict(
            color = 'rgb(221,46,107)',
            outliercolor = 'rgba(224, 35, 79, 0.6)',
            line = dict(
                outliercolor = 'rgba(224, 35, 79, 0.6)',
                outlierwidth = 2)),
)
    
trace_o = go.Box(
    y=y_o,
    x=df_weekly[df_weekly['sender']=='Oliver'].weekday_w.tolist(),
    name='by OJ',
    marker=dict(
        color = 'rgb(0,102,153)',
        outliercolor = 'rgba(0, 73, 153, 0.6)',
        line = dict(
                outliercolor = 'rgba(0, 73, 153, 0.6)',
                outlierwidth = 2)
        )
)
    
layout = go.Layout(
    title='Weekly Messages Breakdown',
    yaxis=dict(
        zeroline=False,
        title='Distinct Messages Sent'
    ),
    boxmode='group'
)

data = [trace_o, trace_g]    
fig2 = go.Figure(data=data, layout=layout)
py.iplot(fig2,filename='analysis2-weekday-msgs')


## time of day analysis:
tot_days = max(df.date_time.dt.date)-min(df.date_time.dt.date)
df_time_day = df.groupby([df['date_time'].dt.hour,'sender']).num_msg.count().reset_index()
df_time_day['avg_mgs_hr'] = df_time_day.num_msg/307

# plotting:
# plot the histogram based on the total counts; 
# on it plot the daily average
gt_daytime = np.array(df_time_day[df_time_day['sender']=='Guy'].avg_mgs_hr)
oj_daytime = np.array(df_time_day[df_time_day['sender']=='Oliver'].avg_mgs_hr)
diff = oj_daytime-gt_daytime

trace1 = go.Bar(
    x=df_time_day[df_time_day['sender']=='Guy'].date_time,
    y=df_time_day[df_time_day['sender']=='Guy'].avg_mgs_hr,
    name='by GT',
    marker=dict(
        color = 'rgb(221,46,107)'
        )
)
trace2 = go.Bar(
    x=df_time_day[df_time_day['sender']=='Oliver'].date_time,
    y=df_time_day[df_time_day['sender']=='Oliver'].avg_mgs_hr,
    name='by OJ',
    marker=dict(
        color='rgb(0,102,153)',
        )
)
trace3 = go.Bar(
    x=df_time_day[df_time_day['sender']=='Oliver'].date_time,
    y=diff,
    name='difference',
    marker=dict(
        color='rgb(244,130,24)',
        )
)

data = [trace1, trace2,trace3]
layout = go.Layout(
    title='Average Hourly Messages',
    xaxis = dict(title='Time of Day'),
    yaxis = dict(title='Average No. of Messages'),
    legend = dict(orientation="h"),
    barmode='group'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='analysis2-avg-per-hour')

# looking at the differences in time

## whos the initiator?
# find the first messages of each day (after 7am)
# find the first messages of each convo (gaps of 2 hours between last and first)

# removing anything before 7am:
df_7_am = df_no_media[df_no_media['date_time'].dt.hour > 6]

# finding the first messages of each day:
df_firsts = df_7_am.groupby(df['date_time'].dt.date).apply(lambda x: x.iloc[[0]])

# plotting:
#1 plot the count of first messages (separate curve pp), group by hour of day, show the timeline
df_firsts = df_firsts.rename(index=str, columns={"date_time": "time1"}).reset_index()
df_firsts['hour'] = df_firsts.time1.dt.hour
#df_firsts = df_firsts.groupby([df_firsts['time1'].dt.hour,'sender']).count()
#df_firsts = df_firsts.drop(columns=['date_time','time1']).reset_index()

trace1 = go.Histogram(
    x=df_firsts[df_firsts.sender=='Guy'].hour,
    name='by GT',
    opacity=0.75,
    xbins=dict(
        start=7.0,
        end=24.0,
        size=1
    ),
    marker=dict(
        color='rgb(221,46,107)',
        )
)
trace2 = go.Histogram(
    x=df_firsts[df_firsts.sender=='Oliver'].hour,
    name='by OJ',
    opacity=0.75,
    xbins=dict(
        start=7.0,
        end=24.0,
        size=1
    ),
    marker=dict(
        color='rgb(0,102,153)',
        )
)

data = [trace2, trace1]
layout = go.Layout(
        barmode='overlay',
        title='First Messages of the Day',
        xaxis = dict(title='Time of Day'),
        yaxis = dict(title = 'Distinct Messages'),
        legend = dict(orientation="h"),
    )
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='analysis2-first-msg-dist')

# 1.5 ploiting number of initiations each month (sum on num-msg, in group by month)
df_firsts.date_time = pd.to_datetime(df_firsts['date_time'])
df_timeline_init = df_firsts.groupby(['sender', df_firsts.date_time.dt.year, df_firsts.time1.dt.month]).sum().reset_index()
from datetime import datetime
df_timeline_init['date'] = df_timeline_init.apply(lambda row: datetime(row['date_time'], row['time1'], 1), axis=1)
#df_timeline_init = df_timeline_init.reindex([3,4,5,6,7,8,9,10,0,1,2,14,15,16,17,18,19,20,21,11,12,13])

traceG1 = go.Scatter(
        x = list(df_timeline_init[df_timeline_init['sender']=='Guy'].date),
        y = list(df_timeline_init[df_timeline_init['sender']=='Guy'].num_msg),
        mode = 'lines',
        name = 'by GT',
        marker = dict(
                color = 'rgb(221,46,107)'
                )
        )

traceO1 = go.Scatter(
        x = list(df_timeline_init[df_timeline_init['sender']=='Oliver'].date),
        y = list(df_timeline_init[df_timeline_init['sender']=='Oliver'].num_msg),
        mode = 'lines',
        name = 'by OJ',
        marker = dict(
                color = 'rgb(0,102,153)'
                )
        )

data = [traceO1, traceG1]
layout = go.Layout(
        title='Initiative Throughout Time',
        xaxis = dict(title = 'Month and Year'),
        yaxis = dict(title = 'Count of days of initiated conversation'),
        legend = dict(orientation="h"),
    )

fig = dict(data=data, layout=layout)
py.iplot(fig, filename='analysis2-initiative-trends')

#2 show the word cloud of the first words

# cleaning stop words from relevant df:
df_firsts["content"] = df_firsts["content"].str.lower().str.split()
# also removing dirty words:
dirt = ['cum','fucking','bottomed']
stop = stopwords.words('english')
stop.extend(dirt)

df_firsts['content'] = df_firsts['content'].apply(lambda x:' '.join([item for item in x if item not in stop]))



ax1 = plt.subplot(121)
# creating word cloud for oliver:
text_oli = df_firsts[df_firsts.sender=='Oliver']['content']
wordcloud_oli = WordCloud(colormap='winter',relative_scaling = 0.69, stopwords = stop, max_font_size=50,background_color="white").generate(' '.join(text_oli))
plt.imshow(wordcloud_oli, interpolation='bilinear')
ax1.set_title('OJs First Messages')
plt.axis("off")

ax2 = plt.subplot(122)
# for guy:
text_guy = df_firsts[df_firsts.sender=='Guy']['content']
wordcloud_guy = WordCloud(colormap='autumn',relative_scaling = 0.69, stopwords = stop, max_font_size=50,background_color="white").generate(' '.join(text_guy))
plt.imshow(wordcloud_guy, interpolation='bilinear')
ax2.set_title('GTs First Messages')
plt.axis("off")
plt.show()


## time delay analysis
df_delay = pd.DataFrame(columns = df.columns)

# filling in a new df with only non-consecutive messages
for i in range(0, df.shape[0]-1):
    if df.iloc[i].sender != df.iloc[i+1].sender:
        df_delay = df_delay.append(df.iloc[i])
    print(i)

# extracting time differences between all messages:
df_delay['diff_dt'] = df_delay.date_time.diff()
df_delay.drop(df_delay.index[0], inplace = True)

df_delay['diff_m'] = df_delay['diff_dt'].dt.total_seconds().div(60).astype(int)
# keeping only gaps between 0 and 3 hours
df_delay = df_delay[df_delay.diff_m < 180]
df_delay = df_delay[df_delay.diff_m >= 0]

# adding day of week:
df_delay['day'] = df_delay.apply (lambda row: label_daysofweek (row),axis=1)

# plotting: violin chart of us one side by side
x = df_delay['day'].tolist()
#y_oli = df_delay[df_delay['sender']=='Oliver'].diff_m.tolist()
#y_guy = df_delay[df_delay['sender']=='Guy'].diff_m.tolist()

import plotly.figure_factory as ff
# NEW OPTION DISTPLOTS

x_gt = df_delay[df_delay['sender']=='Guy'].diff_m
x_oj = df_delay[df_delay['sender']=='Oliver'].diff_m

hist_data = [x_gt, x_oj]
group_labels = ['by GT', 'by OJ']

colors = ['rgb(221,46,107)', 'rgb(0,102,153)']
fig = ff.create_distplot(
    hist_data, group_labels, bin_size=2,colors=colors, show_rug=False
    )
fig['layout'].update(title='Reply Delay Distribution',xaxis=dict(title='Delay [minutes]'))
py.iplot(fig,filename='analysis2-msg-delay-dist')

# OLD OPTION: HISTOGRAM
trace1 = go.Histogram(
    x=df_delay[df_delay['sender']=='Guy'].diff_m,
    name='by GT',
    opacity=0.5,
    xbins=dict(
        start=0.0,
        end=180.0,
        size=2
    ),
    marker=dict(
            color = 'rgb(221,46,107)'
        )
)
trace2 = go.Histogram(
        x=df_delay[df_delay['sender']=='Oliver'].diff_m,
        name='by OJ',
        opacity=0.5,
        xbins=dict(
                start=0.0,
                end=180.0,
                size=2
    ),
    marker=dict(
        color='rgb(0,102,153)',
        )
)

data = [trace1, trace2]
layout = go.Layout(
        barmode='overlay',
        title='Messages Reply Delay (minutes)',
        xaxis = dict(
                title= 'Delay [minutes]'
                ),
        legend = dict(orientation="h"),
    )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='analysis2-msg-delay-hist')

# 2 average daytime delay

df_delay['init_time'] = df_delay.date_time-df_delay.diff_dt
df_delay['init_time'] = df_delay['init_time'].dt.hour
df_av_delay = df_delay.groupby(['init_time', 'sender','day']).mean().reset_index()

df_av_delay_wkday = df_av_delay.groupby(['sender','day']).mean()

df_av_delay_hour = df_av_delay.groupby(['sender','init_time']).mean().reset_index()
# let's bin the delays to be able to do a heatmap:
bins = [0, 2, 5, 10, 30, 60, 120, 180]
labels = ['0-2','2-5','5-10','10-30','30-60','60-120','120-180']

df_delay['binned'] = pd.cut(df_delay['diff_m'], bins=bins, labels=labels)
df_delay_binned = df_delay.groupby(['binned','init_time', 'sender']).count().reset_index()

dataGT = [
    go.Heatmap(
        z=df_delay_binned[df_delay_binned['sender']=='Guy'].content,
        x=df_delay_binned[df_delay_binned['sender']=='Guy'].init_time,
        y=df_delay_binned[df_delay_binned['sender']=='Guy'].binned,
        colorscale='Cividis',
    )
]

layoutGT = go.Layout(
    title='Delay Heatmap, replies by GT',
    xaxis = dict(ticks='', nticks=25, title='Time of day [hr]'),
    yaxis = dict(ticks='', title='Delay [minutes]' )
)

fig = go.Figure(data=dataGT, layout=layoutGT)
py.iplot(fig, filename='analysis2-heatmap-delays-gt')

dataOJ = [
    go.Heatmap(
        z=df_delay_binned[df_delay_binned['sender']=='Oliver'].content,
        x=df_delay_binned[df_delay_binned['sender']=='Oliver'].init_time,
        y=df_delay_binned[df_delay_binned['sender']=='Oliver'].binned,
        colorscale='Cividis',
    )
]

layoutOJ = go.Layout(
    title='Delay Heatmap, replies by OJ',
    xaxis = dict(ticks='', nticks=25, title='Time of day [hr]'),
    yaxis = dict(ticks='', title='Delay [minutes]' )
)

fig = go.Figure(data=dataOJ, layout=layoutOJ)
py.iplot(fig, filename='analysis2-heatmap-delays-oj')

''' dont use:
traceGT = go.Bar(
    x=df_av_delay_hour[df_av_delay_hour.sender=='Guy'].hour,
    name='by GT',
    opacity=0.75,
    xbins=dict(
        start=7.0,
        end=24.0,
        size=1
    ),
    marker=dict(
        color='rgb(221,46,107)',
        )
)
traceOJ = go.Histogram(
    x=df_firsts[df_firsts.sender=='Oliver'].hour,
    name='by OJ',
    opacity=0.75,
    xbins=dict(
        start=7.0,
        end=24.0,
        size=1
    ),
    marker=dict(
        color='rgb(0,102,153)',
        )
)

data = [trace2, trace1]
layout = go.Layout(
        barmode='overlay',
        title='First Messages of the Day',
        legend = dict(orientation="h"),
    )
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='analysis2-first-msg-dist')

'''

### CONTENT

# who sends more media? 
df_media = df[df.content == '<Media omitted>']

count_o = df_media[df_media.sender == 'Oliver'].count() #711
count_g = df_media[df_media.sender == 'Guy'].count() #414

# content overall
df_texts = df_no_media.copy()
df_texts['word_count'] = df_texts['content'].str.count(' ') + 1

word_count_g = df_texts[df_texts['sender']=='Guy'].word_count
tot_words_g = np.sum(word_count_g)
av_g = np.mean(word_count_g)
st_g = np.std(word_count_g)
word_count_o = df_texts[df_texts['sender']=='Oliver'].word_count
tot_words_o = np.sum(word_count_o)
av_o = np.mean(word_count_o)
st_o = np.std(word_count_o)

df_texts['content'] = df_texts['content'].str.lower().str.split()

# creating over all word clouds
stop = stopwords.words('english')

new_stops = ["i'll","yeah","I'll","yeah","yes","no","might","course",
             "I'm","could","I'm gonna","Oh","well","Maybe","don't","it",
             "you're","didn't","thank","you","cum","fuck","bottomed",]
stop.extend(new_stops)

# getting word lists for export:
df_texts['lowercase'] = df_texts["content"].str.lower().str.split()

df_texts['cleaned'] = df_texts['lowercase'].apply(lambda x:' '.join([item for item in x if item not in stop]))




ax1 = plt.subplot(121)
# creating word cloud for oliver:
text_oli = df_texts[df_texts.sender=='Oliver']['content']
wordcloud_oli = WordCloud(colormap='winter',relative_scaling = 0.69, stopwords = stop, max_font_size=50,background_color="white").generate(' '.join(text_oli))
plt.imshow(wordcloud_oli, interpolation='bilinear')
ax1.set_title('OJs Keywords')
plt.axis("off")

ax2 = plt.subplot(122)
# for guy:
text_guy = df_texts[df_texts.sender=='Guy']['content']
wordcloud_guy = WordCloud(colormap='autumn',relative_scaling = 0.69, stopwords = stop, max_font_size=50,background_color="white").generate(' '.join(text_guy))
plt.imshow(wordcloud_guy, interpolation='bilinear')
ax2.set_title('GTs Keywords')
plt.axis("off")
plt.show()

# INtroducing snetiment analysis, using the Vader library:

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()



new_words = {
    'merci': 1.0,
    'merciiii': 2.5,
    'merci bcp': 2.0,
    'boker tov!': 3.0,
    'omg': 1.0,
    ':)))': 5.0,
    'woohoo': 5.0,
    "ain't": -1.5,
    'hahahah': 5.0,
    'hahaha': 5.0,
    'yayyy': 6.0,
    ':))': 4.5,
    'dunno': -1.5,
    'sowwy': -0.5,
    '!!': 2.0,
    'bar': -3.4,
    'tired': -4.5,
    'sick': -5.0,
    'dying': -4.0    
}

analyser.lexicon.update(new_words)

    
def apply_sentiment(sentence):
    snt = analyser.polarity_scores(sentence)
    return snt['neu'],snt['pos'],snt['neg']

df_texts.content = df_texts.content.str.lower()

#pos1,neu1,neg1 = apply_sentiment(content_guy[1])
df_texts['sentiments'] = df_texts.content.apply(lambda sentence: apply_sentiment(sentence))
df_texts[['neutral','positive', 'negative']] = pd.DataFrame(df_texts['sentiments'].tolist(), index=df_texts.index)
df_texts['sentiment_final'] = df_texts[['positive','neutral','negative']].idxmax(axis=1)

# counting sentiments for each:
# Guy:
gt_counts = df_texts[df_texts['sender']=='Guy'].groupby('sentiment_final').count()

oj_counts = df_texts[df_texts['sender']=='Oliver'].groupby('sentiment_final').count()


# pie chart for each:
fig = {
    'data': [
        {
            'labels': ['Positive', 'Negative', 'Neutral'],
            'values': [gt_counts.iloc[2,0], gt_counts.iloc[0,0], gt_counts.iloc[1,0]],
            'type': 'pie',
            'name': 'GT Messages Sentiment',
            'marker': {'colors': ['rgb(62, 216, 134)',
                                  'rgb(252, 85, 113)',
                                  'rgb(167, 173, 178)']},
            'domain': {'x': [0, .49]},
            'hoverinfo':'label+value+name',
            'textinfo':'percent',
            'title': 'Messages by GT'
        },
        {
            'labels': ['Positive', 'Negative', 'Neutral'],
            'values': [oj_counts.iloc[2,0], oj_counts.iloc[0,0], oj_counts.iloc[1,0]],
            'type': 'pie',
            'name': 'OJ Messages Sentiment',
            'marker': {'colors': ['rgb(62, 216, 134)',
                                  'rgb(252, 85, 113)',
                                  'rgb(167, 173, 178)']},
            'domain': {'x': [.50, 1]},
            'hoverinfo':'label+value+name',
            'textinfo':'percent',
            'title': 'Messages by OJ'

        }
            ],
    'layout': {'title': 'Sentiment Analysis of All Messages',
               'showlegend': True}
}
            
py.iplot(fig, filename='analysis2-senitment-piechart')

## daytimr sentiment

df_sent_day = df_texts.groupby(['sender','sentiment_final',df_texts.date_time.dt.hour]).count()
df_sent_day = df_sent_day.drop(df_sent_day.columns[0], axis=1)
df_sent_day = df_sent_day.drop(df_sent_day.columns[1:], axis=1).reset_index()
df_sent_day = df_sent_day[df_sent_day['sentiment_final']!='neutral']
df_sent_day = df_sent_day[df_sent_day['date_time']>6]

df_sent_day_g = df_sent_day[df_sent_day['sender']=='Guy']
df_sent_day_o = df_sent_day[df_sent_day['sender']=='Oliver']
snt_counts_gt = []
snt_counts_oj = []


for i in range(7,24):
    snt_counts_gt.append([int(df_sent_day_g[np.logical_and(df_sent_day_g.date_time==i,df_sent_day_g.sentiment_final=='negative')].content),int(df_sent_day_g[np.logical_and(df_sent_day_g.date_time==i,df_sent_day_g.sentiment_final=='positive')].content)])
    snt_counts_oj.append([int(df_sent_day_o[np.logical_and(df_sent_day_o.date_time==i,df_sent_day_o.sentiment_final=='negative')].content),int(df_sent_day_o[np.logical_and(df_sent_day_o.date_time==i,df_sent_day_o.sentiment_final=='positive')].content)])


data = [
    go.Bar(
        x=df_sent_day_g[df_sent_day_g['sentiment_final']=='negative'].date_time, # assign x as the dataframe column 'x'
        y=df_sent_day_g[df_sent_day_g['sentiment_final']=='negative'].content,
        name='Negative',
        marker = dict(
                color = 'rgb(252, 85, 113)'
                )
    ),
    go.Bar(
        x=df_sent_day_g[df_sent_day_g['sentiment_final']=='positive'].date_time, # assign x as the dataframe column 'x'
        y=df_sent_day_g[df_sent_day_g['sentiment_final']=='positive'].content,
        name='Positive',
        marker = dict(
                color = 'rgb(62, 216, 134)'
                )
    )
]

layout = go.Layout(
    barmode='stack',
    title='Sentiment by Time of Day, GT',
    xaxis = dict(title='Time of Day'),
    yaxis = dict(title= 'No. of Messages')
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='analysis2-daytime-sentiment_g')

data = [
    go.Bar(
        x=df_sent_day_o[df_sent_day_o['sentiment_final']=='negative'].date_time, # assign x as the dataframe column 'x'
        y=df_sent_day_o[df_sent_day_o['sentiment_final']=='negative'].content,
        name='Negative',
        marker = dict(
                color = 'rgb(252, 85, 113)'
                )
    ),
    go.Bar(
        x=df_sent_day_o[df_sent_day_o['sentiment_final']=='positive'].date_time, # assign x as the dataframe column 'x'
        y=df_sent_day_o[df_sent_day_o['sentiment_final']=='positive'].content,
        name='Positive',
        marker = dict(
                color = 'rgb(62, 216, 134)'
                )
    )
]

layout = go.Layout(
    barmode='stack',
    title='Sentiment by Time of Day, OJ',
    xaxis = dict(title='Time of Day'),
    yaxis = dict(title= 'No. of Messages')
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='analysis2-daytime-sentiment_o')

## Adding a table

trace = go.Table(
    header=dict(values=['Variable','Sent by OJ', 'Sent by GT'],
                line = dict(color='#7D7F80'),
                fill = dict(color='#a1c3d1'),
                align = ['left'] * 5),
    cells=dict(values=[['Total messages','msg/day','words/mgs','Photos/videos'],
                       ['22,589', '80+-85','5.0+-4.2', '711'],
                       ['17,408', '60+-59','5.8+-5.1', '414']],
               line = dict(color='#7D7F80'),
               fill = dict(color='#EDFAFF'),
               align = ['left'] * 5))

layout = dict(width=500, height=300)
data = [trace]
fig = dict(data=data, layout=layout)
py.iplot(fig, filename = 'analysis2-table-stats')