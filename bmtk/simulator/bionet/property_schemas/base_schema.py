from bmtk.simulator.bionet import nrn

class CellTypes:
    Biophysical = 0
    Point = 1
    Virtual = 2
    Unknown = 3

    @staticmethod
    def len():
        return 4


class PropertySchema(object):

    #######################################
    # For nodes/cells properties
    #######################################
    def get_cell_type(self, node_params):
        raise NotImplementedError()

    def get_positions(self, node_params):
        raise NotImplementedError()

    def model_type(self, node):
        raise NotImplementedError()

    def get_params_column(self):
        raise NotImplementedError()

    def load_cell_hobj(self, node):
        model_type = self.model_type(node)
        cell_fnc = nrn.py_modules.cell_model(model_type)
        return cell_fnc(node)

    #######################################
    # For edge/synapse properties
    #######################################
    def get_edge_weight(self, src_node, trg_node, edge):
        raise NotImplementedError()

    def preselected_targets(self):
        raise NotImplementedError()

    def target_sections(self, edge):
        raise NotImplementedError()

    def target_distance(self, edge):
        raise NotImplementedError()

    def nsyns(self, edge):
        raise NotImplementedError()

    def load_synapse_obj(self, edge, section_x, section_id):
        synapse_fnc = nrn.py_modules.synapse_model(edge['set_params_function'])
        return synapse_fnc(edge['dynamics_params'], section_x, section_id)
