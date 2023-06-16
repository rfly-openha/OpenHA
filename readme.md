# Something about the OpenHA

The OpenHA (Open Health Assessment) is an easy-to-use and open health assessment framework consisting of various tools and methods.
The framework is designed to make it easier and quicker for users to build and implement their own health assessment applications.

This project is initially launched by the Reliable Flight Control Group (rfly) of Beihang University, and the tentative idea of OpenHA could trace back to the year 2019, just before the breakout of the COVID-19 epidemic.

In the past years, we have done a lot of work to get us prepared to launch this long-term project.

We have done some relative projects, conducted in-depth research, and published some scientific papers.

In a word, our team is growing all the time.

It’s time now.

This project is being developed right now and it’s expected to be officially released before July.

Please contact us if you are also interested in it and would like to join us to make your contributions.

Documents are available in [this repository](https://rfly-openha.github.io/documents/) (not finished yet 😂).

More information about rfly is available on our [official website](http://rfly.buaa.edu.cn/).

E-mail: qq_buaa@buaa.edu.cn

## How to build the project to get the whl file

1. Create a new Python script file named as `setup.py` with following code.

```python
# -*- coding: utf-8 -*-
from setuptools import setup

# dependencies
INSTALL_REQS = [
    'scipy',
    'pandas',
    'matplotlib',
    'requests',
    "tensorflow; platform_system!='Darwin' or platform_machine!='arm64'",
    "tensorflow-macos; platform_system=='Darwin' and platform_machine=='arm64'",
]

# other options
# remember to modify the version when building
setup(
    name='OpenHA',
    version='0.0.3 beta',
    description='The summary description.',
    long_description='The long description',
    long_description_content_type='text/markdown',
    url='https://rfly-openha.github.io',
    author='CuiiGen',
    author_email='cuigen@buaa.edu.cn',
    keywords=[
        'prognostics',
        'diagnostics',
        'fault detection',
        'fdir',
        'physics modeling',
        'prognostics and health management',
        'PHM',
        'health management',
        'surrogate modeling',
        'model tuning',
        'simulation',
        'ivhm',
    ],
    package_dir={"OpenHA": "OpenHA"},
    python_requires='>=3.7, <3.11',
    install_requires=INSTALL_REQS,
)
```

2. Run the following command in cmd. Make sure that the `setup.py` is in the current working dictionary.

```bash
"path_of_python_interpretor" -m build
```

3. The built whl file is in `./dist`.

<!-- OpenHA目前仍需解决的问题 -->
<!-- 以下内容为注释内容，因此为中文 -->

<!-- 卓翼方面的工作

端午节前工作安排：
1. 线上软件的中英文翻译
2. 账号管理
3. https://github.com/rfly-openha/documents 的介绍文档移植过来

项目后续安排：
1. 域名更换或映射
2. 自定义 API 的导入，线上版则需要考虑用户的改动不会对其他人的界面造成影响
3. 考虑是否需要本地版本，本地版如何导入自定义 API
4. 基于某案例的使用文档撰写，简单案例和复杂案例，以及导入 API 操作等其他文字内容 -->

<!-- 我们这边的工作：

1. OpenHA打包发布到PyPi上
2. 目前从Markdown到HTML为手动转换，可以考虑gitbook或者mkdocs（柯博推荐）等框架
3. 继续完善OpenHA的网站以及函数方面的内容
4. 完善例子种类和数量等 -->
