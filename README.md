## LLMFlowGO
An LLM-Driven System for Integrated Workflow Generation and Optimization

<h1>EXAMPLE : Montage workflow</h1>
<p align="center">
<img src="images/LLMWORKFLOW_MONTAGE.png" alt="LLMWORKFLOW_MONTAGE" width="50%"/>
<img src="images/LLM4WORKFLOW_GENERATION.png" alt="LLM4WORKFLOW_GENERATION" width="50%" />
<img src="images/snapshot1.png" alt="snapshot1" width="50%" />
<img src="images/snapshot2.png" alt="snapshot2" width="50%" />
<img src="images/snapshot3.png" alt="snapshot3" width="50%" />
 </p>
 
## ✨ Developer
 * Developer organization:
 1. Shool of Internet, Anhui University, Hefei, China
 2. School of Computer Science and Technology, Anhui University, Hefei, China
 3. School of Information Technology, Deakin University, Geelong, Australia
 4. State Key Lab. for Novel Software Technology, Nanjing University, China

 * Developer Member: 
 * Jia Xu, Weibing Zhang, Xiao Liu,Jinglong Fang, Xuejun Li

## 🤖 Architecture

**Workflow Modeling**: The workflow modeling module uses large language models to automatically convert high-level task descriptions into expressive, adaptable workflow graphs that form the foundation for downstream optimization and execution.This workflow model can be exported from [LLM4Workflow](https://github.com/ISEC-AHU/LLM4Workflow)

**Service Recommendation**: LLMFlowGO distills raw DAGs into a compact semantic payload—built from extracted structures and computed graph features—to filter noise, define feasible service boundaries, and provide clear structural context for stable and efficient edge-environment planning.

**Algorithm Selection and Strategy Identification**: LLMFlowGO conducts a two-stage strategy analysis by first selecting suitable algorithms based on graph semantics, resource heterogeneity, and objectives, and then extracting their core decision logic into constraint-aware semantic interfaces to ensure generated strategies remain verifiable, usable, and system-aligned.

**Workflow Execution, Monitoring, and Optimization**: LLMFlowGO integrates GRS for efficient exploration and MEOH for strategy-driven evaluation, iteratively refining service-range configurations toward a dynamic ideal point and forming a closed feedback loop that enables continuous, quantifiable optimization.These comparison algorithms can be found in [EdgeWorkflow](https://github.com/ISEC-AHU/EdgeWorkflow).
## 🔍 How to use
**We will use the Montage workflow as an example to demonstrate.Montage is a representative astronomy scientific workflow widely adopted as a benchmark for evaluating workflow scheduling and resource management strategies under heterogeneous and data-intensive. environments.
**Step 1:We input the high-level language description of the Montage task into LLM4Workflow.
<p align="center">
<img src="images/LLM4Workflow_step1.png" alt="LLM4Workflow_step1"/>
</p>
Step 2:Modeling the DAG workflow for generating montage.
<p align="center">
<img src="images/LLM4Workflow_step2.png" alt="LLM4Workflow_step2"/>
</p>
Step 3: Select DAG modeling with advanced language generation in LLM4workflow, then you can modify node information as needed, and finally click the Next button.
<p align="center">
<img src="images/step1.png" alt="step1" width="50%" />
</p>
Step 4: LLMFlowGo automatically retrieves the recommended range of services.
<p align="center">
<img src="images/step2.png" alt="step2" width="50%" />
</p>
Step 5: The LLM selects the most suitable algorithm from the external algorithm library based on structural feature information and provides the rationale for the choice. Users can also choose an algorithm according to the actual situation. At the same time, the feature information guides the direction of subsequent generation strategies, and users are allowed to modify the guidance direction as well.
<p align="center">
<img src="images/step3.png" alt="step3" width="50%" />
</p>
Step 6: Workflow execution, monitoring, and optimization in the terminal. In this one-click optimization, the system first allows the LLM to select the functions with the greatest optimization potential based on the algorithm code. Then, it uses the AST to take the relevant function code as a template for subsequent optimizations and extracts the related code into an independent space of the executor. At the same time, it allows extracting the function interfaces called by that function (such as function names, parameters, etc.), enabling the LLM to generate strategies and use the MEOH optimization algorithm, store the optimal code and result comparisons, forming our autonomous system.
<p align="center">
<img src="images/step4.1.png" alt="step4.1" width="50%" />
<img src="images/step4.2.png" alt="step4.2" width="50%" />
</p>

## 🎥 Demonstration

For more details, you can watch the [demo video](https://youtu.be/sq7TER2pDqg).

## 🛠️ Getting Started
To run LLMFlowGo, follow these steps:
1. Clone this repository:`git clone https://github.com/weibinzh/LLMFlowGo.git LLMFlowGo`

2. Set up the Python  environment: `pip install requirements.txt`

3. Start uvicorn server: `python run_server.py`

4. Configure environment variables:

   ```sh
   # Obtain OpenAI API access from https://openai.com/blog/openai-api
   OPENAI_API_KEY=<your-api-key>
   ```
5. Set up the frontend environment: `cd frontend && npm install`

6. Start the frontend (dev) server: `npm run start`

7. Your application should now be up and running in your browser! If you need to change the startup port, you can configure it in the `vite.config.ts` file.
