1. search arxiv https://export.arxiv.org/api/query?search_query=&id_list=2301.09474&sortBy=relevance&sortOrder=descending&start=0&max_results=100
2. search github and choose 5 repo
3. clone the repos
4. gen_code_tree_structure to list repo directory
5. read repo's README.md
6. case_resolved

  ```
  I have determined the reference codebases and paths according to the existing resources and the innovative ideas.
  {
      "reference_codebases": [
          "pygcn",
          "RWKV-LM",
          "graphtransformer",
          "DCRNN_PyTorch",
          "pprgo_pytorch",
          "NodeFormer"
      ],
      "reference_paths": [
          "/workplace/pygcn",
          "/workplace/RWKV-LM",
          "/workplace/graphtransformer",
          "/workplace/DCRNN_PyTorch",
          "/workplace/pprgo_pytorch",
          "/workplace/NodeFormer"
      ],
      "reference_papers": [
          "Semi-Supervised Classification with Graph Convolutional Networks",
          "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention",
          "A Generalization of Transformer Networks to Graphs",
          "Diffusion Convolutional Recurrent Neural Network",
          "Scaling Graph Neural Networks with Approximate PageRank",
          "NodeFormer: A Scalable Graph Structure Learning Transformer for Node Classification"
      ]
  }

  And I have also downloaded the corresponding paper in the Tex format, with the following information:
  Download paper 'Semi-Supervised Classification with Graph Convolutional Networks' successfully
  The paper is downloaded to path: /workplace/papers/semi-supervised_classification_with_graph_convolutional_networks.tex
  Download paper 'Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention' successfully
  The paper is downloaded to path: /workplace/papers/transformers_are_rnns:_fast_autoregressive_transformers_with_linear_attention.tex
  Download paper 'A Generalization of Transformer Networks to Graphs' successfully
  The paper is downloaded to path: /workplace/papers/a_generalization_of_transformer_networks_to_graphs.tex
  Download paper 'Diffusion Convolutional Recurrent Neural Network' successfully
  The paper is downloaded to path: /workplace/papers/diffusion_convolutional_recurrent_neural_network.tex
  Download paper 'Scaling Graph Neural Networks with Approximate PageRank' successfully
  The paper is downloaded to path: /workplace/papers/scaling_graph_neural_networks_with_approximate_pagerank.tex
  Cannot find the paper 'NodeFormer: A Scalable Graph Structure Learning Transformer for Node Classification' in arxiv

  Your task is to do a comprehensive survey on the innovative ideas and the papers, and give me a detailed plan for the implementation.

  Note that the math formula should be as complete as possible, and the code implementation should be as complete as possible. Don't use placeholder code.
  ```

7. wait long time to download papers
8. transfer_to_paper_survey_agent
9. question_answer_on_whole_page('question'='What is the message passing formula in this paper?') for each paper and save paper
10. transfer_back_to_survey_agent
11. plan_dataset
12. Machine Learning Agent -> create directory of project code, write code into right directories and files
13. Machine Learning Agent -> change and debug code
14. Machine Learning Agent -> execute_command('command'='python3 /workplace/project/run_training_testing.py')
15. Code Review Agent -> transfer_to_judge_agent
16. Judge Agent -> 
17. Machine Learning Agent -> execute_command('command'='rm -rf /workplace/project/*')
18. Machine Learning Agent -> create directory and write code with ? times, train the model
19. done Detailed Idea Description