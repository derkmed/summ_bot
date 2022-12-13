# summ_bot

### Set up the conda environment
This will install all the necessary dependencies. Run `conda env create -f environment.yml`

### Augment Daily Dialog with Summaries
Run `python3 -i get_summaries.py`
You can then interactively query for data as follows:
```
>>> aug_data
Dataset({
    features: ['timesteps', 'data', 'sketch', 'summary'],
    num_rows: 1771
})
>>> aug_data['timesteps'][0]
5
>>> aug_data['data'][0]
['Say , Jim , how about going for a few beers after dinner ? ', ' You know that is tempting but is really not good for our fitness . ', ' What do you mean ? It will help us to relax . ', " Do you really think so ? I don't . It will just make us fat and act silly . Remember last time ? ", " I guess you are right.But what shall we do ? I don't feel like sitting at home . ", ' I suggest a walk over to the gym where we can play singsong and meet some of our friends . ', " That's a good idea . I hear Mary and Sally often go there to play pingpong.Perhaps we can make a foursome with them . ", ' Sounds great to me ! If they are willing , we could ask them to go dancing with us.That is excellent exercise and fun , too . ', " Good.Let ' s go now . ", ' All right . ']
>>> for line in aug_data['data'][0][:5]:
...     print(line)
... 
Say , Jim , how about going for a few beers after dinner ? 
 You know that is tempting but is really not good for our fitness . 
 What do you mean ? It will help us to relax . 
 Do you really think so ? I don't . It will just make us fat and act silly . Remember last time ? 
 I guess you are right.But what shall we do ? I don't feel like sitting at home . 
>>> aug_data['sketch'][0]
"0 what about going for a few beers after dinner 1 abstain know that is tempting but is really not good for our fitness 2 none 3 none 4 abstain don't feel like sitting at home 5 none "
>>> aug_data['summary'][0]
" Jim doesn't want to go out for a beer after dinner, because he thinks it"
>>> 
```


