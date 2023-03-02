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

We interact with the environment by applying an action to a specific environment state via `plangym.PlanEnv.step`.
We can define the exact environment state over which we apply our action. 

As expected, this function returns the evolution of the environment,
the observed results, the reward of the performed action, if the agent enters
a terminal state, and additional information about the process.

If we are not interested in getting the current state of the environment, we simply define the argument
`return_state = False` inside the methods `plangym.PlanEnv.reset` and `plangym.PlanEnv.step`: 

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

`plangym` allows applying multiple actions in a single call via the command `plangym.PlanEnv.step_batch`.
The syntax used for this case is reminiscent to that employed when calling a `step` function; we should define 
a __list__ of states and actions and use them as arguments of the function `step_batch()`. `plangym` will
take care of distributing the states and actions correspondingly, returning a tuple with the results
of such actions. 

### Making environments

To initialize an environment, `plangym` uses the same syntax as `gym` via the `plangym.make` command. However, 
this command offers more advanced options than the `gym` standard; it controls the general behavior of the API and
its different environments, and it serves as a command center between the user and the library. 

Instead of using a specific syntax for each environment (with distinct arguments and parameters),
`plangym` unifies all options within a single, common framework under the control of
`make()` command.

All instance attributes are defined through the `make()` command, which classifies and distributes them accordingly whether
they belong to `plangym` or standard parameters. In addition, `make()` also allows the user to configure the
parameters needed for stepping the environment in parallel. One only should select the desired mode, and
`plangym` will do the rest.

```python
import plangym
env = plangym.make(
    name="PlanMontezuma-v0",   # name of the environment
    n_workers=4,  # Number of parallel processes
    state='',  # Define a specific state for the environment
)
```
Once the parameters have been introduced, the command instantiates the appropriate environment class
with the given attributes.


#### Make arguments

`make()` accepts multiple arguments when creating an environment. We should distinguish between the arguments
passed to configurate the environment making process and those used to instantiate the environment itself. 
* Make signature:  
Attributes used to configure the process that creates the environment.
  * `name`: Name of the environment.
  * `n_workers`: Number of workers that will be used to step the environment.
  * `ray`: Use ray for taking steps in parallel when calling `step_batch()`.
  * `domain_name`: Return the name of the agent in the current
  simulation. It is a keyword argument that is only valid for
  `dm_control` environments.
  * `state`: Define a specific state for the environment. The state parameter
  only works for `RetroEnv`, and it is used to select the starting level of the
  selected game. All the other environments do no accept state as a keyword argument,
  and specific states can be set using `get_state()` and `set_state()`.
* Environment instance attributes:
Parameters passed when the class is created. They define and configure the attributes of the class. `make()` accepts
these arguments as _kwargs_. 

All keyword arguments that do not belong to the _Make arguments_ list are passed as _kwargs_ inside `make()`
to instantiate the corresponding environment class (we must emphasize that `plangym` will also use
some attributes included inside the _Make arguments_ classification as instance attributes of the class, such
as `state` or `domain_name`).

#### Instance attributes

As mentioned, users dispose of several parameters to configure the environment creation process
and the attributes of the class itself. Instance parameters are passed as _kwargs_ to the environment class.

Inside these instance attributes, we should differentiate between the attributes managed by `plangym`, and
those that are specific to the `gym` library. `plangym` attributes characterize the envelope that wraps
the original `gym` environment, offering a standard interface among all the processes. `gym` attributes are
those not managed by `plangym` and are passed __directly__ to the `gym.make` method. 

The instance attributes (managed by `plangym`) common to all environment classes are: 
* `name`: Name of the environment. Follows standard gym syntax conventions.
* `frameskip`: Number of times an action will be applied for each ``dt``. When we __step__ the environment,
we take `dt` simulation steps, i.e., we evolve _dt_-times the environment (by applying an action) __in each__
step. Within __each__ simulation step `dt`, we apply the same action `frameskip` times. At the end
of the day, the environment will have evolved `dt * frameskip` times. 
* `autoreset`: Automatically reset the `plangym.environment` when the OpenAI environment returns ``end = True``.
* `wrappers`: Wrappers that will be applied to the underlying OpenAI environment. Every element
of the iterable can be either a class `gym.Wrapper` or a tuple containing ``(gym.Wrapper, kwargs)``.
* `delay_setup`: If ``True``, `plangym` does not initialize the class `gym.environment`and
waits for ``setup`` to be called later. Deferring the environment instantiation gives the users
the option to create it in external processes or when demanded. This fact allows sending `plangym.environment`
as serializable objects, leaving all the settings already defined and prepared for the user to
instantiate the environment when needed. 
* `remove_time_limit`: If `True`, remove the time limit from the environment.
* `render_mode`: Select how the environment and the observations are represented. Options to be
selected are `[None, "human", "rgb_aray"]`.
* `episodic_life`: If `True`, `plangym` sends a terminal signal after loosing a life.
* `obs_type`: Define how `plangym` calculates the observations. Options to be selected
are `["coords", "rgb", "grayscale", None]`.
* `return_image`: If ``True``, 'plangym' adds an "rgb" key in the `info` dictionary returned by
`plangym.env.step` method. This key contains an RGB representation of the environment state.







