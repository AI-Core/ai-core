# ai-core
Allows you to focus on building applications using the latest models and techniques, rather then building them yourself, by providing them in a simple API.

## Getting started

`pip install ai-core`

## `ai_core.models`

Contains models for different types of tasks.

```
from ai_core.models.gans import GAN

gan = GAN()
gan.fit(pytorch_dataloader)
```

## `ai_core.datasets`

```
from ai_core.datasets import Furniture

furniture_dataset = Furniture(root_dir='./images', download=False)
dataloader = torch.utils.data.DataLoader(furniture_dataset)
```

## Contributing
Have a model that you want to make easy to use? Have an awesome dataset to share? Check out the relevant module for examples of how it's done. Then just make a fork, implement your idea, then make a pull request.
