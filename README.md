<!DOCTYPE html>
<html>



<body class="stackedit">
  <div class="stackedit__html"><h1 id="reconstructing-cellular-signaling-dynamics-using-neural-stochastic-differential-equations">Reconstructing noisy gene regulation dynamics using extrinsic-noise-driven neural stochastic differential equations</h1>
<p>This is the official repository for the paper.</p>
<h2 id="dataset">Dataset</h2>
<p>Datasets can be downloaded from https://drive.google.com/file/d/1TiAKLc78ibClgA0t8vmmVUHR0NE8d4Au/view?usp=sharing and https://drive.google.com/file/d/1gOw_hzkxkI7l9cPdthbUVFCPTSdlrmiG/view?usp=sharing.</p>
<h2 id="requirments">Requirments</h2>
<ul>
<!-- <li>numpy&gt;=1.22.4</li>
<li>pandas&gt;=1.3.5</li>
<li>pytorch-transformers&gt;=1.2.0</li>
<li>scikit-learn&gt;=1.0.2</li>
<li>tiktoken&gt;=0.3.3</li>
<li>tokenizers&gt;=0.11.4</li>
<li>torch&gt;=2.4.1</li> -->
<li>torchcde&gt;=0.2.5</li>
<li>torchdiffeq&gt;=0.2.3</li>
<li>torchsde&gt;=0.2.6</li>
<li>torchvision&gt;=0.19.1</li>
<!-- <li>transformers&gt;=4.18.0</li> -->
</ul>
<h2 id="training">Training</h2>
<p>For Circadian examples:</p>
<pre><code>python circadian.py
</code></pre>
<p>For RPA examples:</p>
<pre><code>python sde_RPA.py
<!-- python {baselines}.py -->
</code></pre>
<p>For examples of nfkb:</p>
<pre><code>python learn_noise.py --batch_hyper 1
python learn_noise.py --batch_hyper 2
python learn_noise.py --batch_hyper 4
python learn_noise.py --batch_hyper 8
python learn_noise.py --batch_hyper 16
python learn_noise.py --batch_hyper 32
python train_nSDE.py --batch_hyper 32
python train_nSDE_3dim.py  --batch_hyper 32
python Test_on_experimental_data.py --batch_hyper 32
python Test_on_experimental_data_3dim.py  --batch_hyper 32
</code></pre>
<h2 id="citation">Citation</h2>
    <pre><code>@article{zhang2025reconstructing,
  title={Reconstructing Noisy Gene Regulation Dynamics Using Extrinsic-Noise-Driven Neural Stochastic Differential Equations},
  author={Zhang, Jiancheng and Li, Xiangting and Guo, Xiaolu and You, Zhaoyi and B{\"o}ttcher, Lucas and Mogilner, Alex and Hoffman, Alexander and Chou, Tom and Xia, Mingtao},
  journal={arXiv preprint arXiv:2503.09007},
  year={2025}
}
</code></pre>
</div>
</body>

</html>
