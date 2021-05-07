🚨 🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧 🚨

Library is under development. API and documentation is unstable.

🚨 🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧  🚧 🚨 

# py-aiger-jani
Python library for translating from a subset of Jani to AIGs.

[![Build Status](https://cloud.drone.io/api/badges/mvcisback/py-aiger-jani/status.svg)](https://cloud.drone.io/mvcisback/py-aiger-jani)
[![Docs](https://img.shields.io/badge/API-link-color)](https://mvcisback.github.io/py-aiger-jani)
[![codecov](https://codecov.io/gh/mvcisback/py-aiger-jani/branch/master/graph/badge.svg)](https://codecov.io/gh/mvcisback/py-aiger-jani)
[![PyPI version](https://badge.fury.io/py/py-aiger-jani.svg)](https://badge.fury.io/py/py-aiger-jani)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


<!-- markdown-toc start - Don't edit this section. Run M-x markdown-toc-generate-toc again -->
**Table of Contents**

- [Installation](#installation)
- [Usage](#usage)

<!-- markdown-toc end -->


# Installation

If you just need to use `aiger_jani`, you can just run:

`$ pip install py-aiger-jani`

For developers, note that this project uses the
[poetry](https://poetry.eustace.io/) python package/dependency
management tool. Please familarize yourself with it and then
run:

`$ poetry install`

# Usage

The main entry points for using this library are the `translate_file`
and `translate_jani` functions which convert `jani` files to `AIG`s
represented using the [`py-aiger`](https://github.com/mvcisback/py-aiger) ecosystem (particularly the
[`py-aiger-coins`](https://github.com/mvcisback/py-aiger) package.


Below is an example usage of the `translate_file` function and
`py-aiger-coins` to perform probabilistic inference.

```python
from aiger_jani import translate_file

circ = translate_file("tests/minimdp.jani")

# Use py-aiger-coins to test Pr(x = y) after 3 steps,
# given that actions applied uniformly randomly.

import aiger_bv as BV
from aiger_coins import infer

x, y = BV.uatom(2, 'main-x'), BV.uatom(2, 'main-y')
query = circ.randomize({'edge': {0: 0.5, 1: 0.5}})
query >>= (x == y).aigbv
query = query.unroll(3, only_last_outputs=True)
print(infer.prob(query))  # Output: 0.3421..
```

The `translate_jani` method is analogous, but takes a jani file parsed
into a dictionary using the `json` module.
