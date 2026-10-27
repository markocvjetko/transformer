from torch.sparse import sampled_addmm


##################################
############## EU_MMLU ###########
##################################

def parse_eu_mmlu_hr(sample, tokenizer):

    question = (
        sample["Question"]
    )
    answers = [
        " " + sample["Choice_0"],
        " " + sample["Choice_1"],
        " " + sample["Choice_2"],
        " " + sample["Choice_3"],
    ]
    correct = sample["Answer"]

    question = [tokenizer._special_vocab["<HR>"]] + tokenizer.tokenize(question, add_special=False)
    answers = [tokenizer.tokenize(a, add_special=False) for a in answers]
    correct = int(correct)

    return question, answers, correct



def parse_eu_mmlu_en(sample, tokenizer):

    question = (
        sample["Question"]
    )
    answers = [
        " " + sample["Choice_0"],
        " " + sample["Choice_1"],
        " " + sample["Choice_2"],
        " " + sample["Choice_3"],
    ]
    correct = sample["Answer"]

    question = [tokenizer._special_vocab["<EN>"]] + tokenizer.tokenize(question, add_special=False)
    answers = [tokenizer.tokenize(a, add_special=False) for a in answers]
    correct = int(correct)

    return question, answers, correct

#######################################
############ BELEBELE #################
#######################################

def parse_belebele_hr(sample, tokenizer):

    question = (
        sample["flores_passage"]
        + "\n"
        + "Pitanje: "
        + sample["question"]
        + "\n"
        # + "A: "
        # + sample["mc_answer1"]
        # + "\n"
        # + "B: "
        # + sample["mc_answer2"]
        # + "\n"
        # + "C: "
        # + sample["mc_answer3"]
        # + "\n"
        # + "D: "
        # + sample["mc_answer4"]
        # + "\n"
        + "Odgovor:"
    )
    answers = [
        " " + sample["mc_answer1"],
        " " + sample["mc_answer2"],
        " " + sample["mc_answer3"],
        " " + sample["mc_answer4"],
    ]
    correct = sample["correct_answer_num"]

    question = [tokenizer._special_vocab["<HR>"]] + tokenizer.tokenize(question, add_special=False)
    answers = [tokenizer.tokenize(a, add_special=False) for a in answers]
    correct = int(correct) - 1

    return question, answers, correct

def parse_belebele_en(sample, tokenizer):

    question = (
        sample["flores_passage"]
        + "\n"
        + "Question: "
        + sample["question"]
        + "\n"
        # + "A: "
        # + sample["mc_answer1"]
        # + "\n"
        # + "B: "
        # + sample["mc_answer2"]
        # + "\n"
        # + "C: "
        # + sample["mc_answer3"]
        # + "\n"
        # + "D: "
        # + sample["mc_answer4"]
        # + "\n"
        + "Answer:"
    )
    answers = [
        " " + sample["mc_answer1"],
        " " + sample["mc_answer2"],
        " " + sample["mc_answer3"],
        " " + sample["mc_answer4"],
    ]
    correct = sample["correct_answer_num"]

    question = [tokenizer._special_vocab["<EN>"]] + tokenizer.tokenize(question, add_special=False)
    answers = [tokenizer.tokenize(a, add_special=False) for a in answers]
    correct = int(correct) - 1

    return question, answers, correct


################################
########### FLORES #############
################################



def parse_flores_bs_en(sample, tokenizer):

    original = sample["sentence_bos_Latn"]
    translation = sample["sentence_eng_Latn"]

    original = (
        [tokenizer._special_vocab["<TRANSLATE>"]]
    + [tokenizer._special_vocab["<BS>"]]
    + tokenizer.tokenize(original, add_special=False)
    + [tokenizer._special_vocab["<EN>"]]
    )

    translation = [tokenizer.tokenize(translation, add_special=False)]
    return original, translation


def parse_flores_en_hr(sample, tokenizer):

    original = sample["sentence_hrv_Latn"]
    translation = sample["sentence_eng_Latn"]

    original = (
        [tokenizer._special_vocab["<TRANSLATE>"]] 
        + [tokenizer._special_vocab["<HR>"]]
        + tokenizer.tokenize(original, add_special=False) 
        + [tokenizer._special_vocab["<EN>"]]
    )
    translation = [tokenizer.tokenize(translation, add_special=False)]
    return original, translation



################################
########### COPA ###############
################################

def parse_copa_en(sample, tokenizer):

    if sample["question"] == "effect":
        connector = "Effect: "
    elif sample["question"] == "cause":
        connector = "Cause: "
        
    premise = (
        sample["premise"]
        + connector
    )
    answers = [
        " " + sample["choice1"],
        " " + sample["choice2"],
    ]
    premise = [tokenizer._special_vocab["<EN>"]] + tokenizer.tokenize(premise, add_special=False)
    answers = [tokenizer.tokenize(a, add_special=False) for a in answers]
    correct = sample["label"]

    return premise, answers, correct

    
def parse_copa_hr(sample, tokenizer):


    if sample["question"] == "effect":
        connector = "Posljedica:"
    elif sample["question"] == "cause":
        connector = "Uzrok:"
        
    premise = (
        sample["premise"]
        + connector
    )
    answers = [
        " " + sample["choice1"],
        " " + sample["choice2"],
    ]

    premise = [tokenizer._special_vocab["<HR>"]] + tokenizer.tokenize(premise, add_special=False)
    answers = [tokenizer.tokenize(a, add_special=False) for a in answers]
    correct = sample["label"]

    return premise, answers, correct
