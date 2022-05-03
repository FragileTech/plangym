## Welcome to Plangym

Plangym is an open source Python library for developing and comparing planning algorithms by providing a 
standard API to communicate between algorithms and environments, as well as a standard set of environments 
compliant with that API.

Furthermore, it provides additional functionality for stepping the environments in parallel, delayed environment
initialization for dealing with environments that are difficult to serialize, compatibility with `gym.Wrappers`, 
and more.

## API for reinforcement learning

OpenAI's `gym` has become the de-facto standard in the research community, `plangym`'s API 
is designed to be as similar as possible to `gym`'s API while allowing to modify the environment state.
`plangym` offers a standard API for reinforcement learning problems with a simple, intuitive interface.\
Users with general knowledge of `gym` syntax will feel comfortable using `plangym`; it uses the
same schema and philosophy of the former, yet `plangym` provides new advanced functionalities beyond `gym`
capabilities. 

## Plangym states 

The principal attribute that characterizes `plangym` and distinguishes it from other libraries is the capacity to
save the current state of the environment. By simply calling `get_state()`, the user is able to store the positions,
attributes, actions, and all necessary information that has led the agent and the environment to their actual state. 
In this way, the user can load a specific configuration of the simulation and continue the process in that
precise state. 

## Getting started

### Stepping an environment

We initialize the environment using the command `plangym.make`, similarly to `gym` syntax. By resetting 
the environment, we get our initial _state_ and _observation_. As mentioned, the fact that the environment
is returning its current state is one of the main `plangym` features; we are able to __get__ and __set__
the precise configuration of the environment in each step as if we were loading and saving the
data of a game. This option allows the user to apply a specific action to an explicit state:

```python
import plangym
env = plangym.make(name="CartPole-v0")
state, obs = env.reset()

state = state.copy()
action = env.action_space.sample()

data = env.step(state=state, action=action)
new_state, observ, reward, end, info = data
```

We interact with the environment by applying an action to a specific environment state via `plangym.Env.step`.
We can define the exact environment state over which we apply our action. 

As expected, this function returns the evolution of the environment,
the observed results, the reward of the performed action, if the agent enters
a terminal state, and additional information about the process.

If we are not interested in getting the current state of the environment, we simply define the argument
`return_state = False` inside the methods `plangym.Env.reset` and `plangym.Env.step`: 

```python
import plangym
env = plangym.make(name="CartPole-v0")  
obs = env.reset(return_state=False)  

action = env.action_space.sample()

data = env.step(action=action, return_state=False)
observ, reward, end, info = data
```

By setting `return_state=False`, neither `reset()` nor `step()` will return the state of the simulation. In this way,
we are obtaining the exact same answers as if we were working in a plain `gym` interface. Thus, `plangym`
provides a complete tool for developing planning projects __as well as__ a general, standard API for reinforcement learning problems.  

### Stepping a batch of states and actions
```python
import plangym
env = plangym.make(name="CartPole-v0")
state, obs = env.reset()

states = [state.copy() for _ in range(10)]
actions = [env.action_space.sample() for _ in range(10)]

data = env.step_batch(states=states, actions=actions)
new_states, observs, rewards, ends, infos = data
```

`plangym` allows applying multiple actions in a single call via the command `plangym.Env.step_batch`.
The syntax used for this case is very similar to that employed when calling a `step` function; we only should define 
a __list__ of states and actions and use them as arguments of the function. `plangym` will take care of 
distributing the states and actions, returning a tuple with the results of such actions. 