import csv
import json
from re import compile
from collections import defaultdict
from textblob import TextBlob


def get_sentiments():
    data = defaultdict(lambda: {
        'sentiment': 0,
        'profanities': 0,
        'lines': 0
    })

    with open('dataset/words.json') as file:
        swears = json.load(file)
        pattern = compile(r'\b(' + '|'.join(swears) + r')\b')

    # Loads the lines file and skips the header
    with open('dataset/SouthPark_Lines.csv') as file:
        reader = csv.reader(file)
        next(reader)

        for title, _, line in reader:
            matches = pattern.findall(line.lower())
            data[title]['profanities'] += len(matches)

            # Creates a TextBlob instance with each line, uses NLTK corpora
            sentiment = TextBlob(line).sentiment.polarity
            data[title]['sentiment'] += sentiment

            data[title]['lines'] += 1

    # Compute average and min/max episode sentiment
    minimal = 0
    maximal = 0

    for title, values in data.items():
        values['sentiment'] = values['sentiment'] / values['lines']
        minimal = min(values['sentiment'], minimal)
        maximal = max(values['sentiment'], maximal)

    with open('dataset/SouthPark_Sentiments.csv', 'w') as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow([
            'Title',
            'Sentiment',
            'Profanities'
        ])

        # Write data rows with normalized sentiment
        for title, values in data.items():
            sentiment = (values['sentiment'] - minimal) / (maximal - minimal)
            writer.writerow([
                title,
                round(sentiment, 2),
                values['profanities']
            ])


if __name__ == '__main__':
    get_sentiments()
