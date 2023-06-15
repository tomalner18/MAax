
class Module():
    '''
        Dummy class outline for "Environment Modules".
    '''
    def build_step(self, env, floor, floor_size):
        '''
            This function allows you to add objects to worldgen floor object.
                You could also cache variables needed for observations or add
                information to the env.metadata dict
            Args:
                env (brax.env): the environment
                floor (worldgen.Floor): square worldgen floor object
                floor_size (float): size of the worlgen floor object
            Returns: True if the the build_step was successful, False if it failed
                e.g. build_step might fail because no valid object placements
                were found.
        '''
        return True

    def cache_step(self, env):
        '''
            Caches environment variables in the module data
            Args:
                env (maax.env): the environment
            Returns: None
        '''
        pass

    def observation_step(self, state):
        '''
            Create any observations specific to this module.
            Args:
                env (maax.env): the environment
                state (maax.state): the state
            Returns: dict of observations
        '''
        return {}
