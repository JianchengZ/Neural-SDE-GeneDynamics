<!DOCTYPE html>
<html>

<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Welcome file</title>
  <link rel="stylesheet" href="https://stackedit.io/style.css" />
</head>

<body class="stackedit">
  <div class="stackedit__html"><h1 id="reconstructing-cellular-signaling-dynamics-using-neural-stochastic-differential-equations">Reconstructing cellular signaling dynamics using neural stochastic differential equations</h1>
<p>This is the official repository for the paper.</p>
<h2 id="dataset">Dataset</h2>
<p>Datasets are from xxx and xxx. They can be downloaded from zzz and zzz.</p>
<h2 id="requirments">Requirments</h2>
<ul>
<li>numpy&gt;=1.22.4</li>
<li>pandas&gt;=1.3.5</li>
<li>pytorch-transformers&gt;=1.2.0</li>
<li>scikit-learn&gt;=1.0.2</li>
<li>tiktoken&gt;=0.3.3</li>
<li>tokenizers&gt;=0.11.4</li>
<li>torch&gt;=2.2.1</li>
<li>torchcde&gt;=0.2.5</li>
<li>torchdiffeq&gt;=0.2.3</li>
<li>torchsde&gt;=0.2.6</li>
<li>torchvision&gt;=0.12.0</li>
<li>tqdm&gt;=4.62.3</li>
<li>transformers&gt;=4.18.0</li>
</ul>
<h2 id="training">Training</h2>
<p>For Circadian examples:</p>
<pre><code>python circadian.py
</code></pre>
<p>For RPA examples:</p>
<pre><code>python sde_RPA.py
python {baselines}.py
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
</div>
</body>

</html>
