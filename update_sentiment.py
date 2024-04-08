import torch
import schedule
from elasticsearch import Elasticsearch
from transformers import AutoTokenizer
from langdetect import detect
import datetime
import time
import json
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

bn_model = torch.load('models/banglabert_buet_sentiment_imb_dataset_and_git_dataset_v2.pth', map_location=device)
en_model = torch.load('models/en_model_sentiment.pth', map_location=device)
bn_tokenizer = AutoTokenizer.from_pretrained('csebuetnlp/banglabert')
en_tokenizer = AutoTokenizer.from_pretrained('lxyuan/distilbert-base-multilingual-cased-sentiments-student')

def get_sentiment_analysis(text_lang, text):
    with torch.no_grad():
        if text_lang == 'bn':
            tokenized_text = bn_tokenizer(text, max_length=512, padding='max_length', truncation= True, return_tensors='pt')
            output = bn_model(**tokenized_text).logits
            prediction = torch.argmax(output)
            if prediction==0:
                return 'Positive', output 
            elif prediction == 1:
                return 'Neutral', output
            else:
                return 'Negative', output
        else:
            tokenized_text = en_tokenizer(text, max_length=512, padding='max_length', truncation= True, return_tensors='pt')
            output = en_model(**tokenized_text).logits
            prediction = torch.argmax(output)
            if prediction==0:
                return 'Positive', output
            elif prediction == 1:
                return 'Neutral', output
            else:
                return 'Negative', output

es = Elasticsearch('http://192.168.12.242:9200')
logging.info(f"Elasticsearch connection status: {es.ping()}")  

def cleaning_data(text):
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r' \s+', '', text)
    text = re.sub(r'\|[^|]*\|[^|]*', '', text)

    lang = detect(text)
    if lang == 'bn':
        text = re.sub(r'[a-zA-Z]', '', text)
        text = re.sub(r'আরও বিস্তারিত জানতে ভিজিট করুন:', '', text)

    return text


def process():  
    index_name = "youtube-search"     #Index name here
    logging.info(f"Connected to Index: {index_name}")
    new_field = 'sentiment_flag'
    new_field_data_type = 'boolean'
  
    if es.indices.get_mapping(index=index_name)[index_name]["mappings"]["properties"].get(new_field) is None:
        n_body = {
            "properties": {
                new_field : {
                    "type": new_field_data_type
                }
            }
        }
        es.indices.put_mapping(index=index_name, body=n_body)
        script = f"ctx._source.{new_field} = 'false'"
        es.update_by_query(
        index=index_name,
        body={
            "query": {"match_all": {}},
            "script": script
            }
        )
        logging.info(f"Sentiment flag created in index: {index_name}")

    body = {
    "query": {"match_all": {}}
    }
    search_result = es.search(index=index_name, body=body)

    items = []

    for result in search_result['hits']['hits']:
        try:
            item = {}
            doc_id = result['_id']
            doc_flag = result['_source']['sentiment_flag']
            doc_text = result['_source']['text']   #Text field name here
            if doc_text is None:
                doc_text = result['_source']['title']
        except:
            continue

        if doc_flag == 'false' and doc_text:
            try:
                clean_text = cleaning_data(doc_text)
                language = detect(clean_text)
                sentiment, logits = get_sentiment_analysis(language, clean_text)
                update_date = datetime.datetime.now()
                item['doc_id'] = doc_id
                item['doc_sentiment_flag'] = 'true'
                item['doc_language'] = language
                item['doc_sentiment'] = sentiment
                item['doc_sentiment_logits'] = logits
                item['doc_update_date'] = update_date
        
                items.append(item)
            except:
                continue
        else:
            continue

    for item in items:
        doc_id = item['doc_id']
        doc_update_date_iso = item['doc_update_date'].isoformat()
        doc_sentiment_logits_list = item['doc_sentiment_logits'].tolist()
        update_data = {
        "script": {
            "source": """
            ctx._source.sentiment_flag = params.doc_sentiment_flag;
            ctx._source.text_language = params.doc_language;
            ctx._source.sentiment = params.doc_sentiment;
            ctx._source.sentiment_logits = params.doc_sentiment_logits;
            ctx._source.sentiment_update_date = params.doc_update_date;
            """,
            "params": {
            "doc_sentiment_flag": item['doc_sentiment_flag'],
            "doc_language": item['doc_language'],
            "doc_sentiment": item['doc_sentiment'],
            "doc_sentiment_logits": doc_sentiment_logits_list,
            "doc_update_date": doc_update_date_iso
                }
            }
        }

        # Convert update_data to JSON string
        update_data_json = json.dumps(update_data)
        try:
            es.update(index=index_name, id=doc_id, body=update_data_json)
            logging.info(f"Document updated successfully: {doc_id}")
        except Exception as e:
            logging.error(f"Error updating document {doc_id}: {e}")

def test_method():
    print("hello")
    print(datetime.datetime.now())

schedule.every(1).minutes.do(process)

while True:
    schedule.run_pending()
    time.sleep(1)