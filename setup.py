from setuptools import setup

setup(
    name='ai_core',
    version='0.0.10',    
    description='Allows you to focus on building applications using the latest models and techniques, rather then building them yourself, by providing them in a simple API.',
    url='https://github.com/AI-Core/ai-core',
    author='Harry Berg',
    author_email='harry@theaicore.com',
    license='MIT',
    packages=[
        'ai_core', 
        'ai_core.datasets', 
        'ai_core.models', 
        'ai_core.models.gans'
    ],
    install_requires=['torch', 'torchvision'],
    keywords = ['AI', 'ML'], 
    classifiers=[
    ],
)