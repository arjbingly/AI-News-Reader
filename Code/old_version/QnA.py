from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

class QuestionAnswering:
    def __init__(self, model_name="bert-large-cased-whole-word-masking-finetuned-squad"):
        # Load the pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    def infer(self, question, context):
        # Tokenize the input question and context
        inputs = self.tokenizer(question, context, return_tensors="pt", truncation=True)

        # Get the model's prediction
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Decode the predicted answer
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = self.tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])

        if answer == '[CLS]':
            answer = "Sorry! I could not find the answer to that in the article."

        return answer


# # Example 1
# context = "My name is Amir and I work at CL in the District of Columbia office."
# question = "Where do I work?"
# answer1 = qa_instance.answer_question(question, context)
# print("Answer:", answer1)
#
# # # Example 2
# context = """
#     Paris (French pronunciation: (About this soundlisten)) is the capital and most
#     populous city of France, with an estimated population of 2,175,601 residents as
#     of 2018,in an area of more than 105 square kilometres (41 square miles).[4] Since
#     the 17th century, Paris has been one of Europe's major centres of finance,
#     diplomacy, commerce, fashion, gastronomy, science, and arts. The City of Paris is
#     the centre and seat of government of the region and province of le-de-France, or
#     Paris Region, which has an estimated population of 12,174,880, or about 18 percent
#     of the population of France as of 2017.[5] The Paris Region had a GDP of 709 billion
#     (808 billion)in 2017.[6] According to the Economist Intelligence Unit Worldwide Cost
#     of Living Survey in 2018, Paris was the second most expensive city in the world,
#     after Singapore and ahead of Zurich, Hong Kong, Oslo, and Geneva.[7]
# """
# question = "What is the GDP of Paris?"
# answer2 = qa_instance.answer_question(question, context)
# print("Answer:", answer2)

#
# from news_fetch import NewsArticle
# url = 'https://www.infoq.com/news/2023/12/aws-titan-image-generator/'
# news = NewsArticle(url)
# qa_instance = QuestionAnswering()
# context = news.article.text
#
# question = "What is titan image generator?"
# answer = qa_instance.answer_question(question, context)
# print("Answer:", answer)


