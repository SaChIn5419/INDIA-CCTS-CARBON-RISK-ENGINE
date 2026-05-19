import networkx as nx
from utils.logger import setup_logger

logger = setup_logger('cbam_algorithms', 'logs/cbam_algorithms.log')

class CBAMCradleToGateEngine:
    def __init__(self):
        """
        Calculates recursive Specific Embedded Emissions (SEE) using a Directed Acyclic Graph (DAG)
        to model precursor supply chains as mandated by the EU CBAM Definitive Regime.
        """
        # Directed graph where edges represent precursor consumption
        self.supply_chain = nx.DiGraph()

    def add_product(self, product_id, a_eg, is_complex=True):
        """
        Adds a product node to the DAG.
        a_eg: Attributed direct and indirect emissions for the product itself (gate-to-gate portion)
        """
        self.supply_chain.add_node(product_id, a_eg=a_eg, is_complex=is_complex)

    def add_precursor(self, product_id, precursor_id, mass_consumed_per_unit):
        """
        Adds a directed edge representing that 'product_id' consumes 'precursor_id'.
        mass_consumed_per_unit: Mass of precursor i (m_i) required to produce 1 unit of the product.
        """
        self.supply_chain.add_edge(precursor_id, product_id, mass=mass_consumed_per_unit)

    def calculate_see(self, product_id):
        """
        Recursively calculates Specific Embedded Emissions (SEE_g).
        Formula: SEE_g = a_eg + sum(m_i * SEE_i)
        """
        if not self.supply_chain.has_node(product_id):
            raise ValueError(f"Product {product_id} not found in supply chain.")

        # Ensure DAG is acyclic (no circular dependencies)
        if not nx.is_directed_acyclic_graph(self.supply_chain):
            raise ValueError("Supply chain graph contains cycles, cannot compute SEE.")

        # If it's a simple good (no precursors), SEE is just its attributed emissions
        precursors = list(self.supply_chain.predecessors(product_id))
        node_data = self.supply_chain.nodes[product_id]

        if not node_data.get('is_complex', False) or len(precursors) == 0:
            return node_data['a_eg']

        # Calculate recursively
        see_g = node_data['a_eg']
        for precursor_id in precursors:
            m_i = self.supply_chain[precursor_id][product_id]['mass']
            see_i = self.calculate_see(precursor_id)
            see_g += (m_i * see_i)

        return see_g

class SEFAScalingEngine:
    """
    Computes the Specific Embedded Free Allocation (SEFA).
    EU producers get free allocations, so importers get a matching reduction to avoid double taxation.
    Formula: SEFA_g,y = CBAM_y * CSCF_y * BM_g
    """
    def __init__(self):
        # Phase-in schedule (CBAM_y). e.g., in 2026, 97.5% free allocation remains.
        self.cbam_factor_schedule = {
            2026: 0.975,
            2027: 0.95,
            2028: 0.90,
            2029: 0.775,
            2030: 0.515,
            2034: 0.0
        }
        # Cross-Sectoral Correction Factor (typically 1.0 or slightly less if cap is breached)
        self.cscf = 1.0

    def calculate_sefa(self, year, product_benchmark):
        """
        Calculates SEFA for a given year and EU ETS product benchmark (BM_g).
        """
        if year not in self.cbam_factor_schedule:
            # Linear interpolation or default for unknown years could go here
            # Default to worst-case (0 free allocation) if beyond 2034
            cbam_y = 0.0 if year > 2034 else self.cbam_factor_schedule.get(2026, 0.975)
        else:
            cbam_y = self.cbam_factor_schedule[year]

        return cbam_y * self.cscf * product_benchmark

class CBAMArticle9DeductionEngine:
    """
    Calculates Effective Carbon Price deduction (Article 9).
    Deductions are strictly monetary, netting out implicit domestic subsidies/free allowances.
    """
    def __init__(self):
        pass

    def calculate_effective_price_paid(self, nominal_carbon_price, total_emissions, free_allowances_qty, refunds_and_comp):
        """
        Calculates the Effective Carbon Price paid per tonne in the third country.
        nominal_carbon_price: e.g., CCTS CCC price in EUR/t
        free_allowances_qty: Emissions sheltered under domestic baselines (in tonnes)
        refunds_and_comp: Indirect cost compensation / monetary refunds received (in EUR)
        """
        if total_emissions <= 0:
            return 0.0

        # Chargeable emissions are only those above the free baseline
        chargeable_emissions = max(0, total_emissions - free_allowances_qty)

        # Gross cost paid
        gross_cost = chargeable_emissions * nominal_carbon_price

        # Net cost paid after refunds
        net_cost = max(0, gross_cost - refunds_and_comp)

        # Effective price per tonne of TOTAL emissions
        effective_price = net_cost / total_emissions
        return effective_price

    def calculate_itmo_cap(self, total_emissions, declared_itmo_tonnes):
        """
        Caps Internationally Transferred Mitigation Outcomes (ITMOs) at 10% of reported emissions.
        """
        max_allowed_itmo = total_emissions * 0.10
        return min(declared_itmo_tonnes, max_allowed_itmo)
