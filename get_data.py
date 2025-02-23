import csv
import json
from re import compile
from textblob import TextBlob


def get_sentiments():
    data = {}

    with open('dataset/words.json') as file:
        swears = json.load(file)
        pattern = compile(r'\b(' + '|'.join(swears) + r')\b')

    # Loads the lines file and skips the header
    with open('dataset/SouthPark_Lines.csv') as file:
        reader = csv.reader(file)
        next(reader)

        for title, _, line in reader:
            if title not in data:
                data[title] = {
                    'Positive': 0,
                    'Neutral': 0,
                    'Negative': 0,
                    'Profanities': 0
                }

            matches = pattern.findall(line.lower())
            if matches:
                data[title]['Profanities'] += len(matches)

            # Creates a TextBlob instance with each line, uses NLTK corpora
            line = TextBlob(line)

            # Checks the polarity and increases the adequate counter
            if line.sentiment.polarity > 0.05:
                data[title]['Positive'] += 1
            elif line.sentiment.polarity > -0.05:
                data[title]['Neutral'] += 1
            else:
                data[title]['Negative'] += 1

    with open('dataset/SouthPark_Sentiments.csv', 'w') as file:
        writer = csv.writer(file)

        # Write the header row
        writer.writerow([
            'Title',
            'Positive',
            'Neutral',
            'Negative',
            'Profanities'
        ])

        # Write data rows
        for title, sentiments in data.items():
            writer.writerow([
                title,
                sentiments['Positive'],
                sentiments['Neutral'],
                sentiments['Negative'],
                sentiments['Profanities']
            ])


if __name__ == '__main__':
    get_sentiments()
