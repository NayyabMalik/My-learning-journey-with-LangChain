from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
llm1 = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  
)

llm2 = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  
)
prompt1_notes=PromptTemplate(
    template="give me about  detail notes about following topic {topic}",
    input_variables=["topic"]

)
prompt2_quiz=PromptTemplate(
    template="give me about  detail quiz about following topic {topic}",
    input_variables=["topic"]

)
prompt3_merge=PromptTemplate(
    template="merge the following notes-->{notes} and quizes-->{quizes} into single document",
    input_variables=["notes","quizes"]

)

chat_model1 = ChatHuggingFace(llm=llm1)
chat_model2 = ChatHuggingFace(llm=llm2)



parser=StrOutputParser()


chain1=prompt1_notes|chat_model1|parser
chain2=prompt2_quiz|chat_model2|parser

parallel_chain=RunnableParallel({
    "notes":chain1,"quizes":chain2
})

merge_chain=prompt3_merge|chat_model1|parser
chain=parallel_chain|merge_chain
text="""
A black hole is an astronomical object with a gravitational pull so strong that nothing, not even light, can escape it. A black hole’s “surface,” called its event horizon, defines the boundary where the velocity needed to escape exceeds the speed of light, which is the speed limit of the cosmos. Matter and radiation fall in, but they can’t get out.

Two main classes of black holes have been extensively observed. Stellar-mass black holes with three to dozens of times the Sun’s mass are spread throughout our Milky Way galaxy, while supermassive monsters weighing 100,000 to billions of solar masses are found in the centers of most big galaxies, ours included.

Astronomers had long suspected an in-between class called intermediate-mass black holes, weighing 100 to more than 10,000 solar masses. While a handful of candidates have been identified with indirect evidence, the most convincing example to date came on May 21, 2019, when the National Science Foundation’s Laser Interferometer Gravitational-wave Observatory (LIGO), located in Livingston, Louisiana, and Hanford, Washington, detected gravitational waves from a merger of two stellar-mass black holes. This event, dubbed GW190521, resulted in a black hole weighing 142 Suns.

A stellar-mass black hole forms when a star with more than 20 solar masses exhausts the nuclear fuel in its core and collapses under its own weight. The collapse triggers a supernova explosion that blows off the star’s outer layers. But if the crushed core contains more than about three times the Sun’s mass, no known force can stop its collapse to a black hole. The origin of supermassive black holes is poorly understood, but we know they exist from the very earliest days of a galaxy’s lifetime.

Once born, black holes can grow by accreting matter that falls into them, including gas stripped from neighboring stars and even other black holes.

In 2019, astronomers using the Event Horizon Telescope (EHT) — an international collaboration that networked eight ground-based radio telescopes into a single Earth-size dish — captured an image of a black hole for the first time. It appears as a dark circle silhouetted by an orbiting disk of hot, glowing matter. The supermassive black hole is located at the heart of a galaxy called M87, located about 55 million light-years away, and weighs more than 6 billion solar masses. Its event horizon extends so far it could encompass much of our solar system out to well beyond the planets.
"""
result=chain.invoke({"topic":text})
print(result)
print("CHAIN VISULIZATION")
print(merge_chain.get_graph().print_ascii())