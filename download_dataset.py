import gdown

def download_dataset():

    file_id = "1cOAsKql_xu3L0QgryXrYQTuY5jFvwshQ"
    file_name = "normalized_text.csv"
    gdown.download(id=file_id, output=file_name, quiet=False)