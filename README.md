# gpt_alternatives

## Statistical overview of open large language models in recent years, categorized by base models
| **Model**                             | **#Param**                                                      | **Backbone**       | **Release Date** | **Training Data Source**                                                                                                              | **Training Data Size**    |
|---------------------------------------|-----------------------------------------------------------------|--------------------|------------------|---------------------------------------------------------------------------------------------------------------------------------------|---------------------------|
| T5  (enc-dec)                         | 60M, 220M, 770M, 3B, 11B                                        | Base Model         | 2019-10          | C4                                                                                                                                    | 1T tokens                 |
| mT5  (enc-dec)                        | 300M, 580M, 1.2B, 3.7B, 13B                                     | Base Model         | 2020-10          | mC4                                                                                                                                   | 1T tokens                 |
| GPT-Neo                               | 125M, 350M, 1.3B, 2.7B                                          | Base Model         | 2021-03          | the Pile                                                                                                                              | 825GB                     |
| GPT-NeoX                              | 20B                                                             | Base Model         | 2022-02          | the Pile                                                                                                                              | 825GB                     |
| GPT-J                                 | 6B                                                              | Base Model         | 2021-06          | the Pile                                                                                                                              | 825GB                     |
| OPT                                   | 125M, 1.3B, 2.7B, 6.7B, 13B, 30B, 66B, 175B                     | Base Model         | 2022-05          | the Pile                                                                                                                              | 180B tokens               |
| BLOOM                                 | 560M, 1.1B, 1B7, 3B, 7.1B, 176B                                 | Base Model         | 2022-07          | ROOTS corpus                                                                                                                          | 366B tokens               |
| BLOOMZ                                | 560M, 1.1B, 1B7, 3B, 7.1B, 176B                                 | BLOOM              | 2022-11          | xP3(extended from P3 )                                                                                                                | -                         |
| GLM                                   | 110M, 335M, 410M, 515M, 2B, 10B, 130B                           | Base Model         | 2021-03          | BooksCorpus  and                                                                                                                      |                           |
| English Wikipedia                     | -                                                               |                    |                  |                                                                                                                                       |                           |
| GLM-130B                              | 130B                                                            | Base Model         | 2022-08          | -                                                                                                                                     | -                         |
| ChatGLM                               | 6B                                                              | GLM                | 2023-03          | -                                                                                                                                     | -                         |
| ChatGLM2                              | 6B                                                              | GLM                | 2023-06          | -                                                                                                                                     | -                         |
| LLaMA                                 | 7B, 13B, 33B, 65B                                               | Base Model         | 2023-02          | English CommonCrawl, C4 , Github, Wikipedia, Gutenberg Books3 and Stack Exchange                                                      | 1.4T tokens               |
| OpenLLaMA                             | 3B, 7B                                                          | Replicate of LLaMA | 2023-05          |                                                                                                                                       |                           |
| Alpaca                                | 7B                                                              | LLaMA              | 2023-03          | data generated from text-davinci-003                                                                                                  | 52K                       |
| Vicuna                                | 7B, 13B                                                         | LLaMA              | 2023-03          | user-shared conversations from ShareGPT                                                                                               | 70K                       |
| StableVicuna                          | 13B                                                             | LLaMA \            | Vicuna           | 2023-04                                                                                                                               | -                         |
| BAIZE                                 | 7B, 13B, 30B                                                    | LLaMA              | 2023-04          | dialogs from Quora, StackOverFlow and MedQuAD questions                                                                               | 54K/57K/47K               |
| Koala                                 | 13B                                                             | LLaMA              | 2023-04          | A gather of ShareGPT\footnote{\url{https://sharegpt.com}}, HC3                                                                        | -                         |
| WizardLM                              | 7B, 13B, 30B                                                    | LLaMA              | 2023-06          | evolved instructions (from ShareGPT)/evolved instructions (from Alpaca  data)                                                         | 250k/70k                  |
| UltraLM                               | 13B                                                             | LLaMA              | 2023-06          | UltraChat                                                                                                                             | -                         |
| Pythia                                | 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, 12B                      | Base Model         | 2023-01          | the Pile /the Pile with deduplication applied                                                                                         | 299.9B tokens/207B tokens |
| Dolly-v2                              | 12B                                                             | Pythia             | 2023-04          | instruction/response finetuning records                                                                                               | \textasciitilde 15k       |
| Openchatkit                           | 7B                                                              | Pythia             | 2023-03          | the OIG\footnote{\url{https://laion.ai/blog/oig-dataset/}} dataset                                                                    |                           |
| BELLE-7B                              | 7B                                                              | Pythia             | 2023-03          | a Chinese Dataset\footnote{\url{https://github.com/LianjiaTech/BELLE/tree/main/data/1.5M}}                                            | 1.5M                      |
| StableLM-Alpha                        | 3B, 7B                                                          | Base Model         | 2023-04          | dataset that build on the Pile                                                                                                        | 1.5T tokens               |
| StableLM-Tuned-Alpha                  | 7B                                                              | StableLM           | 2023-04          | Stanford's Alpaca, Nomic-AI's gpt4all, RyokoAI's ShareGPT-52K datasets \cite{sharegpt52K}, Databricks labs' Dolly, and Anthropic's HH | -                         |
| RWKV                                  | 169M, 430M, 1.5B, 3B, 7B, 14B                                   | Base Model         | -                | the Pile                                                                                                                              | 825GB                     |
| ChatRWKV                              | 7B, 14B                                                         | RWKV               | 2022-12          | -                                                                                                                                     | -                         |
| moss-moon-003-base                    | 16B                                                             | base model         | 2023-04          | -                                                                                                                                     | 700B tokens               |
| moss-moon-003-sft                     | 16B                                                             | moss-moon-003-base | 2023-04          | multi-round conversational data                                                                                                       | 1.1 million               |
| RedPajama-INCITE                      | 3B, 7B                                                          | Base Model         | 2023-05          | RedPajama-Data                                                                                                                        | 1.2T tokens               |
| MPT-7B                                | 7B                                                              | Base Model         | 2023-05          | -                                                                                                                                     | 1T tokens                 |
| MPT-7B-Chat                           | 7B                                                              | MPT-7B             | 2023-05          | ShareGPT-Vicuna, HC3, Alpaca, Helpful and Harmless, and Evol-Instruct datasets                                                        | -                         |
| Falcon LLM                            | 7B, 40B                                                         | Base Model         | 2023-06          | -                                                                                                                                     | 1T tokens                 |
| InternLM                              | 7B                                                              | Base Model         | 2023-06          | -                                                                                                                                     | trillions of tokens       |
| InternLM Chat                         | 7B                                                              | InternLM           | 2023-06          | -                                                                                                                                     | -                         |
| Baichuan                              | 7B                                                              | Base Model         | 2023-06          | -                                                                                                                                     | 1.2T tokens               |
| LLAMA 2                               | 7B, 13B, 70B                                                    | Base Model         | 2023-07          | a mix of data from publicly available sources                                                                                         | 2T tokens                 |
| LLAMA 2-CHAT                          | 7B, 13B, 70B                                                    | LLAMA 2            | 2023-07          | publicly available instruction tuning                                                                                                 |                           |
| data and vendor-based annotation data | 27,540 instruction tuning data, 2,919,326 human preference data |                    |                  |                                                                                                                                       |                           |
| Qwen                                  | 7B                                                              | Base Model         | 2023-08          | -                                                                                                                                     | 2.2T tokens               |
| Qwen-Chat                             | 7B                                                              | Qwen               | 2023-08          | -                                                                                                                                     | -                         |

## ChatGPT Alternatives on Different Applications
| **Software**             | **Backbone**                 | **Url**                                       |
|--------------------------|------------------------------|-----------------------------------------------|
| ChatSonic                | GPT-4                        | https://writesonic.com/chat             |
| Jasper Chat              | GPT 3.5 and others           | https://www.jasper.ai/chat              |
| ChatSonic on Opera       | GPT-4                        | https://writesonic.com/chatsonic-opera  |
| NeevaAI                  | ChatGPT                      | https://neeva.com/                      |
| Copilot                  | Codex                        | https://github.com/features/copilot     |
| Tabnine                  | GPT-2                        | https://www.tabnine.com/                |
| Codewhisperer            | -                            | https://aws.amazon.com/cn/codewhisperer |
| Elsa                     | -                            | https://elsaspeak.com/en                |
| DeepL Write              | -                            | https://www.deepl.com/translator        |
| Elicit                   | -                            | https://elicit.org                      |
| Copilot in Azure Quantum | GPT-4                        | https://quantum.microsoft.com/          |
| CoGram                   | -                            | https://www.cogram.com                  |
| Otter                    | -                            | https://otter.ai                        |
| Chatexcel                | -                            | https://chatexcel.com                  |
| AI Anywhere              | ChatGPT, GPT-4               | https://www.ai-anywhere.com/#/dashboard |
| Replika                  | A model with 774M parameters | https://replika.com                     |
| Character AI             | GPT-4                        | https://beta.character.ai               |
| Poe                      | -                            | https://poe.com                         |
| Botsonic AI chatbot      | GPT-4                        | https://writesonic.com/botsonic         |
| ChatPDF                  | ChatGPT                      | https://www.chatpdf.com                |


##  Overview of Datasets for Large Language Models
| Corpora                      | Size       | Latest updated time | Link                                                                |
|------------------------------|------------|---------------------|---------------------------------------------------------------------|
| BoolQ                        | 15,492     | 2019                | https://github.com/google-research-datasets/boolean-questions |
| Hellaswag                    | ~70k  | 2019                | https://allenai.org/data/hellaswag                            |
| WinoGrande                   | ~44k  | 2019                | https://winogrande.allenai.org                               |
| PIQA                         | ~21k  | 2020                | https://yonatanbisk.com/piqa                                 |
| ARC                          | 7,787      | 2018                | https://allenai.org/data/arc                                  |
| OpenbookQA                   | 5,957      | 2018                | https://allenai.org/data/open-book-qa                         |
| RACE                         | $\sim$100k | 2017                | https://www.cs.cmu.edu/~glai1/data/race                      |
| DROP                         | $\sim$96k  | 2019                | https://allenai.org/data/drop                                 |
| GSM8K                        | 8,500      | 2021                | https://github.com/openai/grade-school-math                   |
| MMLU\cite{hendryckstest2021} | 15,908     | 2021                | https://github.com/hendrycks/test      
