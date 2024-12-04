import numpy as np


class Logger(object):
    def __init__(self, save_path):
        self._save_path = save_path
        self._data = dict()
        self._data["meanEnergy"] = []
        self._data["varEnergy"] = []
        self._data["temps"] = []

        self._data["RE_meanEnergy"] = []
        self._data["RE_varEnergy"] = []
        self._data["IM_meanEnergy"] = []
        self._data["IM_varEnergy"] = []

        self._data["meanFreeEnergy"] = []
        self._data["varFreeEnergy"] = []
        self._data["RE_meanFreeEnergy"] = []
        self._data["RE_varFreeEnergy"] = []
        self._data["IM_meanFreeEnergy"] = []
        self._data["IM_varFreeEnergy"] = []

        self._data["training_cost"] = []
        self._data["s_matrix_rank"] = []

        self._data["infidelity"] = []
        self._data["var_infidelity"] = []

        self._data["time_per_step"] = []

    @property
    def save_path(self):
        return self._save_path

    @property
    def data(self):
        return self._data

    def __call__(self, T_np: np.ndarray = None, local_energies_np: np.ndarray = None, log_probs_np: np.ndarray = None,
                 cost_np: np.ndarray = None, s_matrix_rank: np.ndarray = None, infid_np: np.ndarray = None,
                 infid_var_np: np.ndarray = None, time_per_step: np.ndarray = None):
        if local_energies_np is not None:
            self._data["meanEnergy"].append(np.mean(local_energies_np))
            self._data["varEnergy"].append(np.var(local_energies_np))

            self._data["RE_meanEnergy"].append(np.mean(np.real(local_energies_np)))
            self._data["RE_varEnergy"].append(np.var(np.real(local_energies_np)))
            self._data["IM_meanEnergy"].append(np.mean(np.imag(local_energies_np)))
            self._data["IM_varEnergy"].append(np.var(np.imag(local_energies_np)))
        if T_np is not None:
            self._data["temps"].append(T_np)
            Tcomp = T_np + 0.j
            if log_probs_np is not None:
                self._data["meanFreeEnergy"].append(np.mean(local_energies_np + Tcomp * log_probs_np))
                self._data["varFreeEnergy"].append(np.var(local_energies_np + Tcomp * log_probs_np))
                self._data["RE_meanFreeEnergy"].append(np.mean(np.real(local_energies_np + Tcomp * log_probs_np)))
                self._data["RE_varFreeEnergy"].append(np.var(np.real(local_energies_np + Tcomp * log_probs_np)))
                self._data["IM_meanFreeEnergy"].append(np.mean(np.imag(local_energies_np + Tcomp * log_probs_np)))
                self._data["IM_varFreeEnergy"].append(np.var(np.imag(local_energies_np + Tcomp * log_probs_np)))
        if cost_np is not None:
            self._data["training_cost"].append(cost_np)
        if s_matrix_rank is not None:
            self._data["s_matrix_rank"].append(s_matrix_rank)
        if infid_np is not None:
            self._data["infidelity"].append(infid_np)
        if infid_np is not None:
            self._data["var_infidelity"].append(infid_var_np)
        if time_per_step is not None:
            self._data["time_per_step"].append(time_per_step)

    def save(self):
        np.save(self.save_path + '/meanEnergy', np.array(self._data["meanEnergy"]))
        np.save(self.save_path + '/varEnergy', np.array(self._data["varEnergy"]))
        np.save(self.save_path + '/temperatures', np.array(self._data["temps"]))

        np.save(self.save_path + '/RE_meanEnergy', np.array(self._data["RE_meanEnergy"]))
        np.save(self.save_path + '/RE_varEnergy', np.array(self._data["RE_varEnergy"]))
        np.save(self.save_path + '/IM_meanEnergy', np.array(self._data["IM_meanEnergy"]))
        np.save(self.save_path + '/IM_varEnergy', np.array(self._data["IM_varEnergy"]))

        np.save(self.save_path + '/meanFreeEnergy', np.array(self._data["meanFreeEnergy"]))
        np.save(self.save_path + '/varFreeEnergy', np.array(self._data["varFreeEnergy"]))
        np.save(self.save_path + '/RE_meanFreeEnergy', np.array(self._data["RE_meanFreeEnergy"]))
        np.save(self.save_path + '/RE_varFreeEnergy', np.array(self._data["RE_varFreeEnergy"]))
        np.save(self.save_path + '/IM_meanFreeEnergy', np.array(self._data["IM_meanFreeEnergy"]))
        np.save(self.save_path + '/IM_varFreeEnergy', np.array(self._data["IM_varFreeEnergy"]))

        np.save(self.save_path + '/cost', np.array(self._data["training_cost"]))
        if len(self._data["s_matrix_rank"]):
            np.save(self.save_path + '/s_matrix_rank', np.array(self._data["s_matrix_rank"]))
        np.save(self.save_path + '/infidelity', np.array(self._data["infidelity"]))
        np.save(self.save_path + '/var_infidelity', np.array(self._data["var_infidelity"]))
        np.save(self.save_path + '/time_per_step', np.array(self._data["time_per_step"]))
        return True

    def restore(self):
        self._data["meanEnergy"] = np.load(self.save_path + '/meanEnergy.npy').tolist()
        self._data["varEnergy"] = np.load(self.save_path + '/varEnergy.npy').tolist()
        self._data["temps"] = np.load(self.save_path + '/temperatures.npy').tolist()

        self._data["RE_meanEnergy"] = np.load(self.save_path + '/RE_meanEnergy.npy').tolist()
        self._data["RE_varEnergy"] = np.load(self.save_path + '/RE_varEnergy.npy').tolist()
        self._data["IM_meanEnergy"] = np.load(self.save_path + '/IM_meanEnergy.npy').tolist()
        self._data["IM_varEnergy"] = np.load(self.save_path + '/IM_varEnergy.npy').tolist()

        self._data["meanFreeEnergy"] = np.load(self.save_path + '/meanFreeEnergy.npy').tolist()
        self._data["varFreeEnergy"] = np.load(self.save_path + '/varFreeEnergy.npy').tolist()
        self._data["RE_meanFreeEnergy"] = np.load(self.save_path + '/RE_meanFreeEnergy.npy').tolist()
        self._data["RE_varFreeEnergy"] = np.load(self.save_path + '/RE_varFreeEnergy.npy').tolist()
        self._data["IM_meanFreeEnergy"] = np.load(self.save_path + '/IM_meanFreeEnergy.npy').tolist()
        self._data["IM_varFreeEnergy"] = np.load(self.save_path + '/IM_varFreeEnergy.npy').tolist()

        self._data["training_cost"] = np.load(self.save_path + '/cost.npy').tolist()
        try:
            self._data["s_matrix_rank"] = np.load(self.save_path + '/s_matrix_rank.npy').tolist()
        except FileNotFoundError:
            pass
        try:
            self._data["infidelity"] = np.load(self.save_path + '/infidelity.npy').tolist()
        except FileNotFoundError:
            pass
        try:
            self._data["infidelity"] = np.load(self.save_path + '/var_infidelity.npy').tolist()
        except FileNotFoundError:
            pass
        try:
            self._data["time_per_step"] = np.load(self.save_path + '/time_per_step.npy').tolist()
        except FileNotFoundError:
            pass

        return True
