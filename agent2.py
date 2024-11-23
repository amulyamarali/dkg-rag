# funtion to get the correct answer and return it 
# function to use the convert the answer to triples
# function to integrate triples into parent KG (appending it to .csv file)

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from transformers import pipeline



def generate_triples(question):
    # Load the sentence triples
    sentence_triples = load_sentence_triples()
    answer_question(question, sentence_triples, llm)


def generate_answer_a2(question, sentence_triples,llm, tokenizer):

    # Tokenize input
    context = """
        Notre Dame is architecturally Catholic in character, with a golden statue of the Virgin Mary atop the Main Building.
        A copper statue of Christ with arms upraised is located in front of the Main Building.
        The Grotto, a Marian place of prayer and reflection, is located behind the Basilica of the Sacred Heart.
        The Grotto at Notre Dame is a replica of the grotto at Lourdes, France.
        Notre Dame has nine student-run media outlets, including three newspapers, a radio, and television station.
        The Scholastic magazine, started in 1876, claims to be the oldest continuous collegiate publication in the U.S.
        The Observer is a daily newspaper at Notre Dame, published independently of the university.
        The Juggler magazine focuses on student literature and artwork and is published twice a year.
        The Dome yearbook is published annually at Notre Dame.
        In 1987, students published Common Sense in response to The Observer's conservative bias.
        In 2003, students created Irish Rover in response to The Observer's liberal bias.
        The College of Engineering at Notre Dame was established in 1920.
        Notre Dame's College of Engineering includes five departments of study.
        The First Year of Studies program at Notre Dame was established in 1962.
        The First Year of Studies program at Notre Dame offers academic advising and learning resources.
        The First Year of Studies program at Notre Dame has been recognized by U.S. News & World Report.
        Notre Dame first offered graduate degrees in 1854-1855.
        In 1924, Notre Dame introduced formal requirements for graduate degrees, including Doctorate (PhD) programs.
        The School of Architecture at Notre Dame offers a Master of Architecture program.
        The Joan B. Kroc Institute for International Peace Studies was founded in 1986.
        The Kroc Institute offers PhD, Master's, and undergraduate degrees in peace studies.
        Notre Dame's Theodore M. Hesburgh Library was completed in 1963.
        The Word of Life mural on the Hesburgh Library is known as 'Touchdown Jesus'.
        Notre Dame has a competitive admissions process.
        In 2015, Notre Dame admitted 3,577 students out of a pool of 18,156.
        In 2015, 39.1% of admitted students were admitted through early action.
        The Princeton Review ranked Notre Dame as the fifth highest 'dream school' for parents in 2007.
        The Review of Politics at Notre Dame was founded in 1939.
        The Lobund Institute at Notre Dame began researching germ-free life in 1928.
        The Lobund Institute became an independent research organization in the 1940s.
        The Lobund Institute was raised to the status of an Institute in 1950.
        Father James Burns became president of Notre Dame in 1919 and brought academic reforms.
        Father James Burns introduced the elective system at Notre Dame.
        Notre Dame added the College of Commerce by 1921, expanding to five colleges and a law school.
        The university president, John Jenkins, hoped Notre Dame would become a preeminent research institution in his inaugural address.
        In 2007, The Princeton Review ranked Notre Dame as the top school for intramural sports.
        Notre Dame has 29 residence halls on campus.
        Notre Dame residence halls are single-sex with 15 male dorms and 14 female dorms.
        There are no traditional fraternities or sororities at Notre Dame.
        Notre Dame's football stadium is used for the championship game of the intramural season.
        Notre Dame is affiliated with the Congregation of Holy Cross.
        Notre Dame celebrates Catholic Mass over 100 times per week.
        Notre Dame has 57 chapels on campus.
        The university's main building was destroyed by a fire in 1879 and rebuilt by fall of that year.
        Father Sorin and Father Corby led the rebuilding of the Main Building after the 1879 fire.
        Notre Dame's Washington Hall hosted plays and musical acts after its opening in 1880.
        The College of Science at Notre Dame began offering courses in 1880.
        Father Julius Nieuwland at Notre Dame performed early work on neoprene in 1931.
        Notre Dame began studying nuclear physics in 1936 with the construction of a nuclear accelerator.
        In 1913, Father John Augustine Zahm and Theodore Roosevelt embarked on an expedition through the Amazon.
        The university has a strong focus on research in family conflict, genome mapping, and marketing trends.
        Notre Dame's study abroad program ranks sixth in the nation for participation.
        The median starting salary of Notre Dame alumni is $55,300.
        Father Joseph Carrier taught that scientific research aligns with the intellectual and moral culture endorsed by the Church.
        Father John Augustine Zahm defended aspects of evolutionary theory in his 1896 book Evolution and Dogma.
        Father Charles O'Donnell's works focused on philosophy, particularly in addressing questions of ethics and morality.
        The university's basketball team has made 22 NCAA Tournament appearances.
        Notre Dame's football team, known as the Fighting Irish, has won 11 national championships.
        The Fighting Irish have had two Heisman Trophy winners.
        Notre Dame has had three consecutive bowl game appearances in 2019-2021.
        The Notre Dame mascot is the leprechaun, a symbol of Irish heritage.
        The Notre Dame stadium was completed in 1930 and has been expanded since then.
        The football rivalry between Notre Dame and USC is the longest-running and most historic rivalry in college sports.
        The Notre Dame women's soccer team has won multiple NCAA championships.
        The Notre Dame rowing team has received NCAA Championships in women's rowing.
        Notre Dame's athletics program is governed by the NCAA.
        Notre Dame is a member of the Atlantic Coast Conference in athletics.
        The Notre Dame athletics program has a successful cross-country and track team.
        The Fighting Irish mascot is one of the most iconic in college sports.
        Marcus Freeman is the current football coach at Notre Dame.
    """

    # tokenizer = AutoTokenizer.from_pretrained(llm)
    # model = AutoModelForQuestionAnswering.from_pretrained(llm)
    # inputs = tokenizer(question, context, return_tensors="pt")

    # # Get model outputs
    # with torch.no_grad():
    #     outputs = model(**inputs)

    #     # Extract the most likely start and end token positions
    #     start_idx = torch.argmax(outputs.start_logits)
    #     end_idx = torch.argmax(outputs.end_logits)

    #     # Decode the tokens back into text
    #     answer = tokenizer.convert_tokens_to_string(
    #         tokenizer.convert_ids_to_tokens(inputs.input_ids[0][start_idx : end_idx + 1])
    #     )
    # return answer

    # Function to split context into chunks of 512 tokens or less
    # def split_into_chunks(context, max_len=512):
    #     tokens = tokenizer.encode(context)
    #     chunks = [tokens[i:i+max_len] for i in range(0, len(tokens), max_len)]
    #     return chunks

    # # Split context
    # context_chunks = split_into_chunks(context)

    # # Process each chunk independently
    # answers = []
    # for chunk in context_chunks:
    #     # Rebuild chunk with special tokens
    #     chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
    #     inputs = tokenizer(
    #         question, 
    #         chunk_text, 
    #         return_tensors="pt", 
    #         padding=True, 
    #         truncation=True, 
    #         max_length=512
    #     )

    #     with torch.no_grad():
    #         outputs = model(**inputs)
    #         start_idx = torch.argmax(outputs.start_logits)
    #         end_idx = torch.argmax(outputs.end_logits)

    #         # Decode answer from tokens
    #         answer = tokenizer.decode(
    #             inputs.input_ids[0][start_idx : end_idx + 1], 
    #             skip_special_tokens=True
    #         )
    #         answers.append(answer)

    # # Combine answers from each chunk (if needed)
    # final_answer = " ".join(answers).strip()

    # return final_answer

    # Combine question and context for the input
    prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

    # Get the response from the LLM
    response = llm.generate([prompt])

    # Extract and return the answer
    answer = response.generations[0][0].text.strip()
    return answer




    # Initialize the pipeline
    # external_llm_pipeline = pipeline(
    #     "text-generation",
    #     model="gpt2",
    #     max_new_tokens=50,  # Allow up to 50 tokens for answers
    #     pad_token_id=50256  # To handle padding gracefully
    # )

    # # Craft a precise prompt
    # prompt = (
    #     f"Answer the following question concisely and accurately.\n"
    #     f"Question: {question}\n"
    #     f"Answer:"
    # )

    # # Generate the response
    # response = external_llm_pipeline(prompt)[0]['generated_text']

    # # Extract and clean the answer
    # answer = response.split("Answer:")[-1].strip()
    # return answer


