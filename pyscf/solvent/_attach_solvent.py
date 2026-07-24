#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
Attach ddCOSMO to SCF, MCSCF, and post-SCF methods.
'''

import copy
import numpy
from scipy import linalg
from pyscf import lib
from pyscf.lib import logger
from functools import reduce
from pyscf import scf



def _for_scf(mf, solvent_obj, dm=None):
    '''Add solvent model to SCF (HF and DFT) method.

    Kwargs:
        dm : if given, solvent does not respond to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''
    if isinstance(mf, _Solvation):
        mf.with_solvent = solvent_obj
        return mf

    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    sol_mf = SCFWithSolvent(mf, solvent_obj)
    name = solvent_obj.__class__.__name__ + mf.__class__.__name__
    return lib.set_class(sol_mf, (SCFWithSolvent, mf.__class__), name)

# 1. A tag to label the derived method class
class _Solvation:
    pass

class SCFWithSolvent(_Solvation):
    _keys = {'with_solvent'}

    def __init__(self, mf, solvent):
        self.__dict__.update(mf.__dict__)
        self.with_solvent = solvent

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, SCFWithSolvent, name_mixin))
        del obj.with_solvent
        return obj

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        self.with_solvent.check_sanity()
        self.with_solvent.dump_flags(verbose)
        return self

    def reset(self, mol=None):
        self.with_solvent.reset(mol)
        return super().reset(mol)

    # Note v_solvent should not be added to get_hcore for scf methods.
    # get_hcore is overloaded by many post-HF methods. Modifying
    # SCF.get_hcore may lead error.

    def get_veff(self, mol=None, dm=None, *args, **kwargs):
        # FIXME: super() here and after might be problematic and need to be
        # fixed in the future. Consider the combination of solvent and QM/MM.
        # Strictly, vhf = self.undo_solvent().get_veff()
        vhf = super().get_veff(mol, dm, *args, **kwargs)
        with_solvent = self.with_solvent
        if not with_solvent.frozen:
            with_solvent.e, with_solvent.v = with_solvent.kernel(dm)
        e_solvent, v_solvent = with_solvent.e, with_solvent.v

        # NOTE: v_solvent should not be added to vhf in this place. This is
        # because vhf is used as the reference for direct_scf in the next
        # iteration. If v_solvent is added here, it may break direct SCF.
        return lib.tag_array(vhf, e_solvent=e_solvent, v_solvent=v_solvent)

    def get_fock(self, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1,
                 diis=None, diis_start_cycle=None,
                 level_shift_factor=None, damp_factor=None, fock_last=None):
        if dm is None: dm = self.make_rdm1()

        # DIIS was called inside super().get_fock. v_solvent, as a function of
        # dm, should be extrapolated as well. To enable it, v_solvent has to be
        # added to the fock matrix before DIIS was called.
        if getattr(vhf, 'v_solvent', None) is None:
            vhf = self.get_veff(self.mol, dm)
        return super().get_fock(h1e, s1e, vhf+vhf.v_solvent, dm, cycle, diis,
                                diis_start_cycle, level_shift_factor, damp_factor,
                                fock_last)

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        if dm is None:
            dm = self.make_rdm1()
        if getattr(vhf, 'e_solvent', None) is None:
            vhf = self.get_veff(self.mol, dm)

        e_tot, e_coul = super().energy_elec(dm, h1e, vhf)
        e_solvent = vhf.e_solvent
        e_tot += e_solvent
        self.scf_summary['e_solvent'] = vhf.e_solvent.real

        if getattr(self.with_solvent, 'method', '').upper() == 'SMD':
            e_cds = self.with_solvent.get_cds()

            if isinstance(e_cds, numpy.ndarray):
                e_cds = e_cds[0]
            e_tot += e_cds
            self.scf_summary['e_cds'] = e_cds
            logger.info(self, f'CDS correction = {e_cds:.15f}')
        logger.info(self, 'Solvent Energy = %.15g', vhf.e_solvent)
        return e_tot, e_coul

    def nuc_grad_method(self):
        from pyscf.solvent.grad.pcm import make_grad_object
        # FIXME: when applying DF after solvent:
        #    mf = mol.RKS().PCM().density_fit().run()
        # The df.grad.rhf.Gradients.kernel is called. The
        # grad.pcm.WithSolventGrad.kernel is not executed.
        return make_grad_object(self)

    Gradients = nuc_grad_method

    def Hessian(self):
        from pyscf.solvent.hessian.pcm import make_hess_object
        return make_hess_object(self)

    def gen_response(self, *args, **kwargs):
        # The response function consists of two parts: the gas-phase and the
        # solvent response. The "vind" computes the gas-phase response.
        # The attribute .equilibrium_solvation controls whether to add the
        # solvents response.
        #
        # "equilibrium_solvation=True" corresponds to a slow process where the
        # solvents are fully relaxed wrt the first order electron density.
        # Vertical excitations in TDDFT are typically a fast process where the
        # solvation is non-equilibrium. The solvent does not fully relax against
        # the first order density, (corresponding to equilibrium_solvation=False).
        #
        # TDDFT are separately handled in the TDSCFWithSolvent class. This
        # response function handles all other response calculations (such as
        # stability analysis, SOSCF, polarizability and Hessian).
        vind = self.undo_solvent().gen_response(*args, **kwargs)
        is_uhf = isinstance(self, scf.uhf.UHF)
        def vind_with_solvent(dm1):
            v = vind(dm1)
            if self.with_solvent.equilibrium_solvation:
                if is_uhf:
                    v += self.with_solvent._B_dot_x(dm1[0]+dm1[1])
                else:
                    v += self.with_solvent._B_dot_x(dm1)
            return v
        return vind_with_solvent

    def stability(self, *args, **kwargs):
        # When computing orbital hessian, the second order derivatives of
        # solvent energy needs to be computed. It is enabled by
        # the attribute equilibrium_solvation in gen_response method.
        # If solvent was frozen, its contribution is treated as the
        # external potential. The response of solvent does not need to
        # be considered in stability analysis.
        with lib.temporary_env(self.with_solvent,
                               equilibrium_solvation=not self.with_solvent.frozen):
            return super().stability(*args, **kwargs)

    def to_gpu(self):
        from gpu4pyscf.solvent import _attach_solvent # type: ignore
        solvent_obj = self.with_solvent.to_gpu()
        obj = _attach_solvent._for_scf(self.undo_solvent().to_gpu(), solvent_obj)
        return obj

    def TDA(self, equilibrium_solvation=False):
        return _for_tdscf(super().TDA(), equilibrium_solvation=equilibrium_solvation)

    def TDHF(self, equilibrium_solvation=False):
        return _for_tdscf(super().TDHF(), equilibrium_solvation=equilibrium_solvation)

    CasidaTDDFT = NotImplemented

    def TDDFT(self, equilibrium_solvation=False):
        return _for_tdscf(super().TDDFT(), equilibrium_solvation=equilibrium_solvation)

    def MP2(self):
        solvent_model = _dispatch_solvent_model(self.with_solvent)
        # Note the super().MP2 might actually point to the DFMP2
        return solvent_model(super().MP2())

    def CISD(self):
        solvent_model = _dispatch_solvent_model(self.with_solvent)
        return solvent_model(super().CISD())

    def CCSD(self):
        solvent_model = _dispatch_solvent_model(self.with_solvent)
        # Note the super().CCSD might actually point to the DFCCSD
        return solvent_model(super().CCSD())

    def CASCI(self, ncas, nelecas, **kwargs):
        solvent_model = _dispatch_solvent_model(self.with_solvent)
        return solvent_model(super().CASCI(ncas, nelecas, **kwargs))

    def CASSCF(self, ncas, nelecas, **kwargs):
        solvent_model = _dispatch_solvent_model(self.with_solvent)
        return solvent_model(super().CASSCF(ncas, nelecas, **kwargs))

def _dispatch_solvent_model(solvent_obj):
    from pyscf import solvent
    solvent_name = solvent_obj.__class__.__name__
    if solvent_name in ('PCM', 'ddCOSMO', 'ddPCM', 'SMD'):
        return getattr(solvent, solvent_name)
    if solvent_name == 'PolEmbed':
        return solvent.PE
    raise RuntimeError(f'Unknown solvent model {solvent}')


# for MC-PDFT
def _for_mcpdft(mc, solvent_obj, dm=None):

    if isinstance(mc, _Solvation):
        mc.with_solvent = solvent_obj
        return mc

    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    sol_cas = MCPDFTWithSolvent(mc, solvent_obj)
    name = solvent_obj.__class__.__name__ + mc.__class__.__name__
    return lib.set_class(sol_cas, (MCPDFTWithSolvent, mc.__class__), name)


class MCPDFTWithSolvent(_Solvation):
    _keys = {'with_solvent'}

    def __init__(self, mc, solvent):
        self.__dict__.update(mc.__dict__)
        self.with_solvent = solvent

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, MCPDFTWithSolvent, name_mixin))
        del obj.with_solvent
        return obj

    # Additional methods for MCPDFT with solvent can be added here
    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        self.with_solvent.check_sanity()
        self.with_solvent.dump_flags(verbose)
        
        return self
    
    def reset(self, mol=None):
        self.with_solvent.reset(mol)
        return super().reset(mol)
    

    def optimize_mcscf_(self, mo_coeff=None, ci0=None, **kwargs):
        '''Optimize the MC-SCF wave function underlying an MC-PDFT calculation.
        Has the same calling signature as the parent kernel method. '''
        

        # redefining the mc_obj here
        from pyscf.mcscf.addons import StateAverageFCISolver
        
        if isinstance(self.fcisolver, StateAverageFCISolver):
            mc_obj = self._mc_class(self._scf, self.ncas, self.nelecas).state_average_(self.weights)
        else:
            mc_obj = self._mc_class(self._scf, self.ncas, self.nelecas)

        mc_obj.__dict__.update(self.__dict__)
        mc_obj = _for_casscf(mc_obj, self.with_solvent, dm=None)    

        mc_obj.__class__ = type(mc_obj.__class__.__name__, (mc_obj.__class__,),
                         {'dump_flags': self.dump_flags(verbose=self.verbose)})

        from pyscf.mcpdft.mcpdft import _mcscf_env
        with _mcscf_env(self):
            self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy = mc_obj.kernel()  

        if isinstance(self.fcisolver, StateAverageFCISolver):
            self._final_state_dms = mc_obj._final_state_dms.copy() # keep it for later use (may be not needed any more)

        return self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    
    def compute_pdft_energy_(self, mo_coeff=None, ci=None, ot=None, otxc=None,
                             grids_level=None, grids_attr=None, dump_chk=True, verbose=None, **kwargs):
        '''Compute the MC-PDFT energy(ies) (and update stored data)
        with the MC-SCF wave function fixed. '''
        
        if mo_coeff is not None: self.mo_coeff = mo_coeff
        if ci is not None: self.ci = ci
        if ot is not None: self.otfnal = ot
        if otxc is not None: self.otxc = otxc
        if grids_attr is None: grids_attr = {}
        if grids_level is not None: grids_attr['level'] = grids_level
        if len(grids_attr): self.grids.__dict__.update(**grids_attr)
        if verbose is None: verbose = self.verbose
        self.verbose = self.otfnal.verbose = verbose
        nroots = getattr(self.fcisolver, 'nroots', 1)


        epdft = [self.energy_tot(mo_coeff=self.mo_coeff, ci=self.ci, state=ix,
                                 logger_tag='MC-PDFT state {}'.format(ix))
                 for ix in range(nroots)]
        self.e_ot = [e_ot for e_tot, e_ot in epdft]
        
        from pyscf.mcscf.addons import StateAverageMCSCFSolver

        logger.note(self,"\n********************** (MC-PDFT+solvent) ************************")

        if isinstance(self, StateAverageMCSCFSolver):
            e_states = [e_tot for e_tot, e_ot in epdft]
            try:
                self.e_states = e_states
            except AttributeError as e:
                self.fcisolver.e_states = e_states
                assert (self.e_states is e_states), str(e)
            # TODO: redesign this. MC-SCF e_states is stapled to
            # fcisolver.e_states, but I don't want MS-PDFT to be
            # because that makes no sense
            self.e_tot = numpy.dot(e_states, self.weights)
            e_states = self.e_states

            g_states = []

            for i, ei in enumerate(e_states):
                g_states.append(ei+self.with_solvent.e[i])
                logger.note(self,'  State %d   E(MC-PDFT+solvent) = %.15g',
                            i, ei+self.with_solvent.e[i])
                
            # quantity update     
            try:
                self.e_states = g_states
            except AttributeError as e:
                self.fcisolver.e_states = g_states
                assert (self.e_states is g_states), str(e)

            self.e_tot = numpy.dot(self.e_states, self.weights)
            e_states = self.e_states

        elif nroots > 1:  # nroots>1 CASCI
            self.e_tot = [e_tot for e_tot, e_ot in epdft]
            e_states = self.e_tot

        else:  # nroots==1 not StateAverage class
            self.e_tot, self.e_ot = epdft[0]
            e_states = [self.e_tot]

            logger.note(self,'  State %d   E(MC-PDFT+solvent) = %.15g',
                        len(e_states)-1, self.e_tot+self.with_solvent.e)
                
            # quantity update
            self.e_tot = self.e_tot + self.with_solvent.e
            e_states = [self.e_tot]

        if dump_chk:
            e_tot = self.e_tot
            e_ot = self.e_ot
            self.dump_chk(locals())

        logger.note(self,"****************************  *  *******************************\n") 
    
        
        return self.e_tot, self.e_ot, e_states



    def to_gpu(self):
        obj = self.undo_solvent().to_gpu()
        obj = _for_mcpdft(obj, self.with_solvent)
        return lib.to_gpu(self, obj)


def _for_casscf(mc, solvent_obj, dm=None):
    '''Add solvent model to CASSCF method.

    Kwargs:
        dm : if given, solvent does not respond to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''

    if isinstance(mc, _Solvation):
        mc.with_solvent = solvent_obj
        return mc
    
    from pyscf.mcscf.addons import StateAverageFCISolver


    if isinstance(mc.fcisolver, StateAverageFCISolver):
        solvent_obj.state_id = None
        

    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    sol_cas = CASSCFWithSolvent(mc, solvent_obj)
    name = solvent_obj.__class__.__name__ + mc.__class__.__name__
    return lib.set_class(sol_cas, (CASSCFWithSolvent, mc.__class__), name)

class CASSCFWithSolvent(_Solvation):
    _keys = {'with_solvent'}

    def __init__(self, mc, solvent):
        self.__dict__.update(mc.__dict__)
        self.with_solvent = solvent
        self._e_tot_without_solvent = 0
        self._e_states_corrected = 0


    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, CASSCFWithSolvent, name_mixin))
        del obj.with_solvent
        del obj._e_tot_without_solvent
        return obj

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        self.with_solvent.check_sanity()
        self.with_solvent.dump_flags(verbose)
        if self.conv_tol < 1e-7:
            logger.info(self, 'CASSCF+solvent may not be able to '
                        'converge to conv_tol=%g', self.conv_tol)

        if (getattr(self._scf, 'with_solvent', None) and
            not getattr(self, 'with_solvent', None)):
            logger.warn(self, '''Solvent model %s was found in SCF object.
Solvent is not applied to the CASCI object. The CASSCF result is not affected by the SCF solvent model.
To enable the solvent model for CASSCF, a decoration to CASSCF object as below needs to be called
    from pyscf import solvent
    mc = mcscf.CASSCF(...)
    mc = solvent.ddCOSMO(mc)
''',
                        self._scf.with_solvent.__class__)
            
        if self.with_solvent.frozen:
            logger.info(self, '\n** The solvent potential is frozen in the CASSCF **\n')
        else:
            logger.info(self, '\n** The solvent potential is updated self-consistently in the CASSCF **\n')


        return self


    def reset(self, mol=None):
        self.with_solvent.reset(mol)
        return super().reset(mol)

    def update_casdm(self, mo, u, fcivec, e_ci, eris, envs={}):

        casdm1, casdm2, gci, fcivec = \
                super().update_casdm(mo, u, fcivec, e_ci, eris, envs)

        
# The potential is generated based on the density of current micro iteration.
# It will be added to hcore in casci function. Strictly speaking, this density
# is not the same to the CASSCF density (which was used to measure
# convergence) in the macro iterations.  When CASSCF is converged, it
# should be almost the same to the CASSCF density of the last macro iteration.

        with_solvent = self.with_solvent
        if not with_solvent.frozen:
                        
            from pyscf.mcscf.addons import StateAverageFCISolver
            if isinstance(self.fcisolver, StateAverageFCISolver):
                _sa_casdms = StateAverageFCISolver.states_make_rdm1(self.fcisolver, fcivec, self.ncas, self.nelecas)
                dm = []
                for _sa_casdm in _sa_casdms:
                    mocore = mo[:,:self.ncore]
                    mocas = mo[:,self.ncore:self.ncore+self.ncas]
                    _sa_1rdm = numpy.dot(mocore, mocore.conj().T) * 2
                    _sa_1rdm = _sa_1rdm + reduce(numpy.dot, (mocas, _sa_casdm, mocas.conj().T))
                    dm.append(_sa_1rdm)
            else:
                # Code to mimic dm = self.make_rdm1(ci=fcivec)
                mocore = mo[:,:self.ncore]
                mocas = mo[:,self.ncore:self.ncore+self.ncas]
                dm = reduce(numpy.dot, (mocas, casdm1, mocas.T))
                dm += numpy.dot(mocore, mocore.T) * 2

            with_solvent.e, with_solvent.v = with_solvent.kernel(dm) 
                                                                 
        return casdm1, casdm2, gci, fcivec

# Solvent Potential should be added to the effective potential. However, there
# is no hook to modify the effective potential in CASSCF. The workaround
# here is to modify hcore. It can affect the 1-electron operator in many CASSCF
# functions: gen_h_op, update_casdm, casci.  Note hcore is used to compute the
# energy for core density (Ecore).  The resultant total energy from casci
# function will include the contribution from ddCOSMO potential. The
# duplicated energy contribution from solvent needs to be removed.
    def get_hcore(self, mol=None):
        hcore = self._scf.get_hcore(mol)
        
        if self.with_solvent.v is not None:
            hcore += self.with_solvent.v    #if mc is constructed with the scf object which has solvent potential
                                            #,then that HF potential is carried over to this mc object,
                                            # and will be added at the 1st CASSCF iteration
                                            # So, the initial guess of with_solvent.v = self._scf.with_solvent.v

        return hcore

    
    def _finalize(self):
        from pyscf.mcscf.addons import StateAverageFCISolver
        if isinstance(self.fcisolver, StateAverageFCISolver):
            try:
                self.e_states = self._e_states_corrected
            except AttributeError as e:
                self.fcisolver.e_states = self._e_states_corrected
                assert (self.e_states is self._e_states_corrected), str(e)
        super()._finalize()
        return self
    
    def casci(self, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
        log = logger.new_logger(self, verbose)
        log.debug('Running CASCI with solvent. Note the total energy '
                  'has duplicated contributions from solvent.')

        # In super().casci function, dE was computed based on the total
        # energy without removing the duplicated solvent contributions.
        # However, envs['elast'] is the last total energy with correct
        # solvent effects. Hack envs['elast'] to make super().casci print
        # the correct energy difference.
        envs['elast'] = self._e_tot_without_solvent

        e_tot, e_cas, fcivec = super().casci(mo_coeff, ci0, eris,
                                             verbose, envs)

        self.mo_coeff = mo_coeff 

        with_solvent = self.with_solvent 
        from pyscf.mcscf.addons import StateAverageFCISolver
        if isinstance(self.fcisolver, StateAverageFCISolver):
            self._e_tot_without_solvent = e_tot
            _sa_casdms = StateAverageFCISolver.states_make_rdm1(self.fcisolver, fcivec, self.ncas, self.nelecas)

            dm = []
            for _sa_casdm in _sa_casdms:
                mocore = self.mo_coeff[:,:self.ncore]
                mocas = self.mo_coeff[:,self.ncore:self.ncore+self.ncas]
                _sa_1rdm = numpy.dot(mocore, mocore.conj().T) * 2
                _sa_1rdm = _sa_1rdm + reduce(numpy.dot, (mocas, _sa_casdm, mocas.conj().T))
                dm.append(_sa_1rdm)

            _e_states = self.e_states.copy()
            self._final_state_dms = dm.copy()

            log.debug('Computing corrections to the total energy.')
            if with_solvent.v is not None:
                edups = []
                for i in range(len(_e_states)):
                    edup = numpy.einsum('ij,ji->', with_solvent.v, dm[i])
                    edups.append(edup)

                if not with_solvent.frozen:
                    e_solv, v_solv = with_solvent.kernel(dm) 
                                                                                                                       
                else:
                    e_solv =  with_solvent.e
                
                for i in range(len(_e_states)):
                    _e_states[i] = _e_states[i] - edups[i] + e_solv[i]
        
                e_tot = numpy.einsum('i,i->', _e_states, self.weights) 

                self._e_states_corrected = _e_states.copy()

                log.info("\n********************** (CASSCF+solvent) ************************")
                log.info('State-averaged E(CASSCF+solvent) = %.15g \n'
                    ,e_tot)
                log.info("Energy for each state:")
                for i, ei in enumerate(_e_states):
                    log.info('  State %d weight %g  E(CASSCF+solvent) = %.15g',
                        i, self.weights[i], ei)
                log.info("****************************  *  *******************************\n")

        else:
            dm = self.make_rdm1(ci=fcivec, ao_repr=True)
            self._e_tot_without_solvent = e_tot

            if with_solvent.v is not None:

                log.debug('Computing corrections to the total energy.')
                edup = numpy.einsum('ij,ji->', with_solvent.v, dm)

                if not with_solvent.frozen:
                    e_solv, v_solv = with_solvent.kernel(dm) 
                                                                                                                 
                else:
                    e_solv =  with_solvent.e

                e_tot = e_tot - edup + e_solv

                
                log.info("\n********************** (CASSCF+solvent) ************************")
                log.info('E(CASSCF+solvent) = %.15g \n'
                        ,e_tot)
                log.info("****************************  *  *******************************\n")
                

        # Update solvent effects for next iteration if needed
        if not with_solvent.frozen:
            with_solvent.e = e_solv
            with_solvent.v = v_solv
        
        return e_tot, e_cas, fcivec

    def nuc_grad_method(self):
        logger.warn(self, '''
The code for CASSCF gradients was based on variational CASSCF wavefunction.
However, the ddCOSMO-CASSCF energy was not computed variationally.
Approximate gradients are evaluated here. A small error may be expected in the
gradients which corresponds to the contribution of
MCSCF_DM * V_solvent[d/dX MCSCF_DM] + V_solvent[MCSCF_DM] * d/dX MCSCF_DM
''')
        from pyscf.solvent.grad.pcm import make_grad_object
        return make_grad_object(self)

    Gradients = nuc_grad_method

    def to_gpu(self):
        obj = self.undo_solvent().to_gpu()
        obj = _for_casscf(obj, self.with_solvent)
        return lib.to_gpu(self, obj)


# for L-PDFT
def _for_lpdft(mc, solvent_obj, dm=None):

    if isinstance(mc, _Solvation):
        mc.with_solvent = solvent_obj
        return mc
    
    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    sol_cas = LPDFTWithSolvent(mc, solvent_obj)
    name = solvent_obj.__class__.__name__ + mc.__class__.__name__
    return lib.set_class(sol_cas, (LPDFTWithSolvent, mc.__class__), name)


class LPDFTWithSolvent(_Solvation):
    _keys = {'with_solvent'}
    _extra_keys = {'lpdft_max_iter', 'lpdft_conv_tol'} 

    def __init__(self, mc, solvent):
        self.__dict__.update(mc.__dict__)
        self.with_solvent = solvent
        self.lpdft_max_iter = getattr(mc, 'lpdft_max_iter', 50)
        self.lpdft_conv_tol = getattr(mc, 'lpdft_conv_tol', 1e-7)


    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, LPDFTWithSolvent, name_mixin))
        del obj.with_solvent
        return obj
    
    def reset(self, mol=None):
        self.with_solvent.reset(mol)
        return super().reset(mol)
    

    # Additional methods for LPDFT with solvent are added here -->
    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        self.with_solvent.check_sanity()
        self.with_solvent.dump_flags(verbose)
        
        return self
    
    def optimize_mcscf_(self, mo_coeff=None, ci0=None, **kwargs):
        '''Optimize the MC-SCF wave function underlying an MC-PDFT calculation.
        Has the same calling signature as the parent kernel method. '''
        

        # redefining the mc_obj here
        from pyscf.mcscf.addons import StateAverageFCISolver
        
        if isinstance(self.fcisolver, StateAverageFCISolver):
            mc_obj = self._mc_class(self._scf, self.ncas, self.nelecas).state_average_(self.weights)
        else:
            mc_obj = self._mc_class(self._scf, self.ncas, self.nelecas)


        mc_obj.__dict__.update(self.__dict__)
        
        mc_obj = _for_casscf(mc_obj, self.with_solvent, dm=None)    

        mc_obj._keys = mc_obj._keys | self._extra_keys

        mc_obj.__class__ = type(mc_obj.__class__.__name__, (mc_obj.__class__,),
                         {'dump_flags': self.dump_flags(verbose=self.verbose)})
        
        from pyscf.mcpdft.mcpdft import _mcscf_env
        with _mcscf_env(self):
            self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy = mc_obj.kernel()  

        if isinstance(self.fcisolver, StateAverageFCISolver):
            self._final_state_dms = mc_obj._final_state_dms.copy() # keep it for later use


        return self.e_mcscf, self.e_cas, self.ci, self.mo_coeff, self.mo_energy


    def make_lpdft_ham_(self, mo_coeff=None, ci=None, ot=None, 
                        vmat=None):
        """Compute the L-PDFT Hamiltonian

        Args:
            mo_coeff : ndarray of shape (nao, nmo)
                A full set of molecular orbital coefficients. Taken from self if
                not provided.

            ci : list of ndarrays of length nroots
                CI vectors should be from a converged CASSCF/CASCI calculation

            ot : an instance of on-top functional class - see otfnal.py

            vmat : ndarray of shape (nao, nao) 
                Solvent potential matrix 
                

        Returns:
            lpdft_ham : ndarray of shape (nroots, nroots) or (nirreps, nroots, nroots)
                Linear approximation to the MC-PDFT energy expressed as a
                hamiltonian in the basis provided by the CI vectors. If
                StateAverageMix, then returns the block diagonal of the lpdft
                hamiltonian for each irrep.
        """
        from pyscf.fci import direct_spin1
        from pyscf.mcpdft.lpdft import _LPDFTMix
        from pyscf.mcpdft import _dms


        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if ci is None:
            ci = self.ci
        if ot is None:
            ot = self.otfnal

    
        ot.reset(mol=self.mol)

        spin = abs(self.nelecas[0] - self.nelecas[1])
        omega, _, hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc, spin=spin)
        if abs(omega) > 1e-11:
            raise NotImplementedError("range-separated on-top functionals")
        if abs(hyb[0] - hyb[1]) > 1e-11:
            raise NotImplementedError(
                "hybrid functionals with different exchange, correlations components"
            )

        cas_hyb = hyb[0]


        ncas = self.ncas

        casdm1s_0, casdm2_0 = self.get_casdm12_0(ci=ci) 
        

        self.veff1, self.veff2, E_ot = self.get_pdft_veff(
            mo=mo_coeff,
            ci=ci, 
            casdm1s=casdm1s_0,
            casdm2=casdm2_0,
            drop_mcwfn=True,
            incl_energy=True,
            ot=ot
        )

        # This is all standard procedure for generating the hamiltonian in PySCF
        h1, h0 = self.get_h1lpdft(E_ot, casdm1s_0, casdm2_0, hyb=1.0 - cas_hyb, mo_coeff=mo_coeff,
                                  vmat=vmat) 
        h2 = self.get_h2lpdft()
        h2eff = direct_spin1.absorb_h1e(h1, h2, ncas, self.nelecas, 0.5)

        def construct_ham_slice(solver, slice, nelecas):
            ci_irrep = ci[slice]
            if hasattr(solver, "orbsym"):
                solver.orbsym = self.fcisolver.orbsym

            hc_all_irrep = [solver.contract_2e(h2eff, c, ncas, nelecas) for c in ci_irrep]
            lpdft_irrep = numpy.tensordot(ci_irrep, hc_all_irrep, axes=((1, 2), (1, 2)))
            diag_idx = numpy.diag_indices_from(lpdft_irrep)
            lpdft_irrep[diag_idx] += h0 + cas_hyb * self.e_mcscf[slice]


            return lpdft_irrep

        if not isinstance(self, _LPDFTMix):
            return construct_ham_slice(direct_spin1, slice(0, len(ci)), self.nelecas)

        # We have a StateAverageMix Solver
        self._irrep_slices = []
        start = 0
        for solver in self.fcisolver.fcisolvers:
            end = start + solver.nroots
            self._irrep_slices.append(slice(start, end))
            start = end

        return [
            construct_ham_slice(s, irrep, self.fcisolver._get_nelec(s, self.nelecas))
            for s, irrep in zip(self.fcisolver.fcisolvers, self._irrep_slices)
        ]

    def get_lpdft_hconst(
        self,
        E_ot,
        casdm1s_0,
        casdm2_0,
        hyb=1.0,
        ncas=None,
        ncore=None,
        veff1=None,
        veff2=None,
        mo_coeff=None,
    ):
        """Compute h_const for the L-PDFT Hamiltonian

        Args:
        self : instance of class _PDFT

        E_ot : float
            On-top energy

        casdm1s_0 : ndarray of shape (2, ncas, ncas)
            Spin-separated 1-RDM in the active space generated from expansion
            density.

        casdm2_0 : ndarray of shape (ncas, ncas, ncas, ncas)
            Spin-summed 2-RDM in the active space generated from expansion
            density.

        Kwargs:
        hyb : float
            Hybridization constant (lambda term)

        ncas : float
            Number of active space MOs

        ncore: float
            Number of core MOs

        veff1 : ndarray of shape (nao, nao)
            1-body effective potential in the AO basis computed using the
            zeroth-order densities.

        veff2 : pyscf.mcscf.mc_ao2mo._ERIS instance
            Relevant 2-body effective potential in the MO basis.

        neq : bool
            If True, use optical dielectric K_sym_opt.
            If False, use static dielectric K_sym.

        q_slow : ndarray of shape (ngrids,) or None
            Slow PCM charges from ground state. Only used when neq=True.

        v_grids_0 : ndarray of shape (ngrids,) or None
            Total molecular ESP at grid points from ground state.
            Only used when neq=True.

        Returns:
            Constant term h_const for the expansion term.
        """

        from pyscf.mcpdft import _dms

        if ncas is None:
            ncas = self.ncas
        if ncore is None:
            ncore = self.ncore
        if veff1 is None:
            veff1 = self.veff1
        if veff2 is None:
            veff2 = self.veff2
        if mo_coeff is None:
            mo_coeff = self.mo_coeff

        nocc = ncore + ncas

        # Get the 1-RDM matrices
        casdm1_0 = casdm1s_0[0] + casdm1s_0[1]
        dm1s = _dms.casdm1s_to_dm1s(self, casdm1s=casdm1s_0, mo_coeff=mo_coeff)
        dm1 = dm1s[0] + dm1s[1]

        # Coulomb interaction
        vj = self._scf.get_j(dm=dm1)
        e_veff1_j = numpy.tensordot(veff1 + hyb * 0.5 * vj, dm1)

        # Deal with 2-electron on-top potential energy
        e_veff2 = veff2.energy_core
        e_veff2 += numpy.tensordot(veff2.vhf_c[ncore:nocc, ncore:nocc], casdm1_0)
        e_veff2 += 0.5 * numpy.tensordot(
            veff2.papa[ncore:nocc, :, ncore:nocc, :], casdm2_0, axes=4
        )
        # h_nuc + E_ot - 1/2 g_pqrs D_pq D_rs - V_pq D_pq - 1/2 v_pqrs d_pqrs
        energy_core = hyb * self.energy_nuc() + E_ot - e_veff1_j - e_veff2


        return energy_core

    def get_lpdft_hcore_only(self, casdm1s_0, hyb=1.0, mo_coeff=None,
                          ncore=None, ncas=None, vmat=None):
        """
        Returns the lpdft hcore AO integrals weighted by the
        hybridization factor. Excludes the MC-SCF (wfn) component.

        Kwargs:
        vmat : ndarray of shape (nao, nao) 
                Solvent potential matrix 

        """
        if vmat is None:
            vmat = self.with_solvent.v

        from pyscf.mcpdft import _dms

        dm1s = _dms.casdm1s_to_dm1s(self, casdm1s=casdm1s_0, mo_coeff=mo_coeff,
                                    ncore=ncore, ncas=ncas)
        dm1 = dm1s[0] + dm1s[1]
        v_j = self._scf.get_j(dm=dm1)
        h_eff = hyb * self._scf.get_hcore() + self.veff1 + hyb * v_j 

        h_eff += vmat  # add the solvent potential
        return h_eff
    
    def get_lpdft_hcore(self, casdm1s_0=None, mo_coeff=None, ncore=None, ncas=None, vmat=None):
        """
        Returns the full lpdft hcore AO integrals. Includes the MC-SCF
        (wfn) component for hybrid functionals.

        Kwargs:
        vmat : ndarray of shape (nao, nao) 
                Solvent potential matrix 
        """

        if vmat is None:
            vmat = self.with_solvent.v


        if casdm1s_0 is None:
            casdm1s_0 = self.get_casdm12_0()[0]

        spin = abs(self.nelecas[0] - self.nelecas[1])
        cas_hyb = self.otfnal._numint.rsh_and_hybrid_coeff(self.otfnal.otxc, spin=spin)[
            2
        ]
        hyb = 1.0 - cas_hyb[0]

        return cas_hyb[0] * self._scf.get_hcore() + self.get_lpdft_hcore_only(
            casdm1s_0, hyb=hyb, mo_coeff=mo_coeff, ncore=ncore, ncas=ncas, vmat=vmat
        )
    
    def transformed_h1e_for_cas(
        self, E_ot, casdm1s_0, casdm2_0, hyb=1.0, mo_coeff=None, ncas=None, ncore=None,
        vmat=None,
    ):
        """Compute the CAS one-particle L-PDFT Hamiltonian

        Args:
            mc : instance of a _PDFT object

            E_ot : float
                On-top energy

            casdm1s_0 : ndarray of shape (2,ncas,ncas)
                Spin-separated 1-RDM in the active space generated from expansion
                density

            casdm2_0 : ndarray of shape (ncas,ncas,ncas,ncas)
                Spin-summed 2-RDM in the active space generated from expansion
                density

            hyb : float
                Hybridization constant (lambda term)

            mo_coeff : ndarray of shape (nao,nmo)
                A full set of molecular orbital coefficients. Taken from self if
                not provided.

            ncas : int
                Number of active space molecular orbitals

            ncore : int
                Number of core molecular orbitals

            vmat : ndarray of shape (nao, nao) 
                The solvent potential matrix. 



        Returns:
            A tuple, the first is the effective one-electron linear PDFT
            Hamiltonian defined in CAS space, the second is the modified core
            energy.
        """
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if ncas is None:
            ncas = self.ncas
        if ncore is None:
            ncore = self.ncore

        nocc = ncore + ncas
        mo_core = mo_coeff[:, :ncore]
        mo_cas = mo_coeff[:, ncore:nocc]

        # h_pq + V_pq + J_pq all in AO integrals
        hcore_eff = self.get_lpdft_hcore_only(casdm1s_0, hyb=hyb, mo_coeff=mo_coeff,
                                        ncore=ncore, ncas=ncas,
                                        vmat=vmat)
        energy_core = self.get_lpdft_hconst(E_ot, casdm1s_0, casdm2_0, hyb,
                                      mo_coeff=mo_coeff, ncore=ncore,
                                      ncas=ncas)

        if mo_core.size != 0:
            core_dm = numpy.dot(mo_core, mo_core.conj().T) * 2
            # This is precomputed in MRH's ERIS object
            energy_core += self.veff2.energy_core
            energy_core += numpy.tensordot(core_dm, hcore_eff).real

        h1eff = mo_cas.conj().T @ hcore_eff @ mo_cas
        # Add in the 2-electron portion that acts as a 1-electron operator
        h1eff += self.veff2.vhf_c[ncore:nocc, ncore:nocc]

        return h1eff, energy_core
    
    def get_h1lpdft(self, E_ot, casdm1s_0, casdm2_0, hyb=1.0, mo_coeff=None,
                    vmat=None):
        return self.transformed_h1e_for_cas(
            E_ot, casdm1s_0, casdm2_0, hyb=hyb, mo_coeff=mo_coeff,
            vmat=vmat)
    
    def _get_K_sym_opt(self):
        """Build K^opt and
        Cache in _intermediates."""
        with_solvent = self.with_solvent
        if 'K_sym_opt' not in with_solvent._intermediates:
            K_opt, R_opt = with_solvent._get_K_opt_R_opt()
            K_inv_R_opt = numpy.linalg.solve(K_opt, R_opt)
            with_solvent._intermediates['K_sym_opt'] = 0.5 * (K_inv_R_opt + K_inv_R_opt.T)
        return with_solvent._intermediates['K_sym_opt']

    def get_solv_dms(self):
        from pyscf.mcscf.addons import StateAverageFCISolver
        _sa_casdms = StateAverageFCISolver.states_make_rdm1(self.fcisolver, self.ci, self.ncas, self.nelecas)

        dm = []
        for _sa_casdm in _sa_casdms:
            mocore = self.mo_coeff[:,:self.ncore]
            mocas = self.mo_coeff[:,self.ncore:self.ncore+self.ncas]
            _sa_1rdm = numpy.dot(mocore, mocore.conj().T) * 2
            _sa_1rdm = _sa_1rdm + reduce(numpy.dot, (mocas, _sa_casdm, mocas.conj().T))
            dm.append(_sa_1rdm)

        return dm
        
    def kernel(self, mo_coeff=None, ci0=None, ot=None, verbose=None, dump_chk=True):

        from pyscf.mcpdft import _dms

        if ot is None:
            ot = self.otfnal
        ot.reset(mol=self.mol)
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff

        log = logger.new_logger(self, verbose)
        if ci0 is None and isinstance(getattr(self, "ci", None), list):
            ci0 = [c.copy() for c in self.ci]

        # SA-CASSCF+PCM optimization
        self.optimize_mcscf_(mo_coeff=mo_coeff, ci0=ci0)
        ci_mcscf = [c.copy() for c in self.ci]

        with_solvent = self.with_solvent
        dms = self.get_solv_dms()
        _, vmat = with_solvent.kernel(dms)
        with_solvent.v = vmat
        
        lpdft_conv = False
        _iter = 0
        e_tot_prev = 0.0

        # recompute solvent potential from L-PDFT states densities until convergence
        log.info("\nStarting L-PDFT+solvent iterations...")
        while not lpdft_conv and _iter < self.lpdft_max_iter:            

            self.lpdft_ham = self.make_lpdft_ham_(
                ot=ot, ci=ci_mcscf, vmat=vmat)  

            if hasattr(self, "_irrep_slices"):
                e_states, si_pdft = zip(*map(self._eig_si, self.lpdft_ham))
                e_states = numpy.concatenate(e_states)
                si_pdft  = linalg.block_diag(*si_pdft)
            else:
                e_states, si_pdft = self._eig_si(self.lpdft_ham)
                        
            self.si_pdft  = si_pdft
            self.ci = list(numpy.tensordot(si_pdft.T, numpy.asarray(ci_mcscf), axes=1))

            # energy corrections
            dm = self.get_solv_dms()
            edups = []
            for i in range(len(e_states)):
                edup = numpy.einsum('ij,ji->', vmat, dm[i])
                edups.append(edup)
        
            e_solv, v_solv = with_solvent.kernel(dm) 
            for i in range(len(e_states)):
                e_states[i] = e_states[i] - edups[i] + e_solv[i]

            self.e_states = e_states
            self.e_tot    = numpy.dot(e_states, self.weights)

            lpdft_dE =abs(self.e_tot - e_tot_prev)
            _iter += 1
            log.info(f"** Solvent self-consistent cycle {_iter}:\n dE = {lpdft_dE}")
            self._finalize_lin()

            if (_iter >= self.lpdft_max_iter) and (lpdft_dE > self.lpdft_conv_tol):
                log.info("self-consistent L-PDFT+solvent not converged within the maximum number of iterations \n"
                "Increase mc.lpdft_max_iter.")

            if (lpdft_dE <= self.lpdft_conv_tol):
                lpdft_conv = True
                log.info("self-consistent L-PDFT+solvent converged")
                
            e_tot_prev = self.e_tot
            vmat = v_solv
            with_solvent.v = v_solv

        return (self.e_tot, self.e_mcscf, self.e_cas,
                self.ci, self.mo_coeff, self.mo_energy)


    def _finalize_lin(self):
        log = logger.Logger(self.stdout, self.verbose)
        nroots = len(self.e_states)
        log.info("\n********************** (L-PDFT+solvent) ************************")
        if log.verbose >= logger.NOTE and getattr(self.fcisolver, "spin_square", None):
            ss = self.fcisolver.states_spin_square(self.ci, self.ncas, self.nelecas)[0]


            for i in range(nroots):
                log.note(
                    "  State %d weight %g  E(LPDFT+solvent) = %.15g  S^2 = %.7f",
                    i,
                    self.weights[i],
                    self.e_states[i],
                    ss[i],
                )

        else:
            for i in range(nroots):
                log.note(
                    "  State %d weight %g  E(LPDFT+solvent) = %.15g",
                    i,
                    self.weights[i],
                    self.e_states[i],
                )    
        log.info("****************************  *  *******************************\n")

    def to_gpu(self):
        obj = self.undo_solvent().to_gpu()
        obj = _for_lpdft(obj, self.with_solvent)
        return lib.to_gpu(self, obj)



def _for_casci(mc, solvent_obj, dm=None):
    '''Add solvent model to CASCI method.

    Kwargs:
        dm : if given, solvent does not respond to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''
    if isinstance(mc, _Solvation):
        mc.with_solvent = solvent_obj
        mc.dm = dm
        return mc

    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    sol_mc = CASCIWithSolvent(mc, solvent_obj, dm)
    name = solvent_obj.__class__.__name__ + mc.__class__.__name__
    return lib.set_class(sol_mc, (CASCIWithSolvent, mc.__class__), name)

class CASCIWithSolvent(_Solvation):
    _keys = {'with_solvent', 'dm'}

    def __init__(self, mc, solvent, dm=None):
        self.__dict__.update(mc.__dict__)
        self.with_solvent = solvent
        self.dm = dm

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, CASCIWithSolvent, name_mixin))
        del obj.with_solvent

        if hasattr(obj, 'dm'):
            del obj.dm

        return obj

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        self.with_solvent.check_sanity()
        self.with_solvent.dump_flags(verbose)

        if self.with_solvent.frozen:
            logger.info(self, '\n** The solvent potential is frozen in the CASCI **')

        return self

    def reset(self, mol=None):
        self.with_solvent.reset(mol)
        return super().reset(mol)

    def get_hcore(self, mol=None):
        hcore = self._scf.get_hcore(mol)
        if self.with_solvent.v is not None:
            # NOTE: get_hcore was called by CASCI to generate core
            # potential.  v_solvent is added in this place to take accounts the
            # effects of solvent. Its contribution is duplicated and it
            # should be removed from the total energy.
            hcore += self.with_solvent.v
        return hcore

    def kernel(self, mo_coeff=None, ci0=None, verbose=None):
        with_solvent = self.with_solvent

        log = logger.new_logger(self)
        log.info('\n** Self-consistently update the solvent effects for %s **',
                 self.__class__.__name__)
        log1 = copy.copy(log)
        log1.verbose -= 1  # Suppress a few output messages

        mc_base_kernel = super().kernel
        def casci_iter_(ci0, log):
            # self.e_tot, self.e_cas, and self.ci are updated in the call
            # to super().kernel
            e_tot, e_cas, ci0 = mc_base_kernel(mo_coeff, ci0, log)[:3]


            if isinstance(self.e_cas, (float, numpy.number)):
                dm = self.make_rdm1(ci=ci0)
            else:
                log.note('Computing solvent responses to DM of state %d',
                          with_solvent.state_id)
                dm = self.make_rdm1(ci=ci0[with_solvent.state_id])

                if with_solvent.e is not None:
                    edup = numpy.einsum('ij,ji->', with_solvent.v, dm) 
                    self.e_tot += with_solvent.e - edup


            self.e_tot = e_tot.copy()
            if not with_solvent.frozen:
                with_solvent.e, with_solvent.v = with_solvent.kernel(dm)
            return self.e_tot, e_cas, ci0

        if with_solvent.frozen:
            with lib.temporary_env(self, _finalize=lambda:None):
                casci_iter_(ci0, log)

            log.info("\n********************** (CASCI+solvent) ************************")
            log.info("Energy for each state:")
            for i, ei in enumerate(self.e_tot):
                log.info('  State %d E(CASCI+solvent) = %.15g',
                        i, ei)
            log.info("****************************  *  *******************************\n")
            
            return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

        self.converged = False
        with lib.temporary_env(self, canonicalization=False):
            e_tot = e_last = 0
            for cycle in range(self.with_solvent.max_cycle):
                log.info('\n** Solvent self-consistent cycle %d:', cycle)
                e_tot, e_cas, ci0 = casci_iter_(ci0, log1)

                de = e_tot - e_last
                if isinstance(e_cas, (float, numpy.number)):      
                    log.info('Solvent cycle %d  E(CASCI+solvent) = %.15g  '
                                'dE = %g', cycle, e_tot, de)
                else:
                    for i, e in enumerate(e_tot):
                        log.info('Solvent cycle %d  CASCI root %d  '
                                'E(CASCI+solvent) = %.15g  dE = %g',
                                cycle, i, e, de[i])
                            
                if abs(e_tot-e_last).max() < with_solvent.conv_tol:
                    self.converged = True
                    break
                e_last = e_tot

        # An extra cycle to canonicalize CASCI orbitals
        with lib.temporary_env(self, _finalize=lambda:None):
            casci_iter_(ci0, log)
        if self.converged:
            log.info('self-consistent CASCI+solvent converged')
        else:
            log.info('self-consistent CASCI+solvent not converged')
        log.note('Total energy with solvent effects')
        self._finalize()
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    def nuc_grad_method(self):
        logger.warn(self, '''
The code for CASCI gradients was based on variational CASCI wavefunction.
However, the ddCOSMO-CASCI energy was not computed variationally.
Approximate gradients are evaluated here. A small error may be expected in the
gradients which corresponds to the contribution of
MCSCF_DM * V_solvent[d/dX MCSCF_DM] + V_solvent[MCSCF_DM] * d/dX MCSCF_DM
''')
        from pyscf.solvent.grad.pcm import make_grad_object
        return make_grad_object(self)

    Gradients = nuc_grad_method

    def to_gpu(self):
        obj = self.undo_solvent().to_gpu()
        obj = _for_casci(obj, self.with_solvent)
        return lib.to_gpu(self, obj)


def _for_post_scf(method, solvent_obj, dm=None):
    '''A wrapper of solvent model for post-SCF methods (CC, CI, MP etc.)

    NOTE: this implementation often causes (macro iteration) convergence issue

    Kwargs:
        dm : if given, solvent does not respond to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''
    if isinstance(method, _Solvation):
        method.with_solvent = solvent_obj
        method._scf.with_solvent = solvent_obj
        return method

    # Ensure that the underlying _scf object has solvent model enabled
    if getattr(method._scf, 'with_solvent', None):
        scf_with_solvent = method._scf
    else:
        scf_with_solvent = _for_scf(method._scf, solvent_obj, dm)
        if dm is None:
            solvent_obj = scf_with_solvent.with_solvent
            solvent_obj.e, solvent_obj.v = \
                    solvent_obj.kernel(scf_with_solvent.make_rdm1())

    if dm is not None:
        solvent_obj = scf_with_solvent.with_solvent
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    postmf = PostSCFWithSolvent(method, scf_with_solvent)
    name = solvent_obj.__class__.__name__ + method.__class__.__name__
    return lib.set_class(postmf, (PostSCFWithSolvent, method.__class__), name)

class PostSCFWithSolvent(_Solvation):
    def __init__(self, method, scf_with_solvent):
        self.__dict__.update(method.__dict__)
        self._scf = scf_with_solvent
        # Post-HF objects access the solvent effects indirectly through the
        # underlying ._scf object.
        self._basic_scanner = method.as_scanner()
        self._basic_scanner._scf = scf_with_solvent.as_scanner()

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self._scf.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, PostSCFWithSolvent, name_mixin))
        obj._scf = self._scf.undo_solvent()
        del obj._basic_scanner
        return obj

    @property
    def with_solvent(self):
        return self._scf.with_solvent

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        self.with_solvent.check_sanity()
        self.with_solvent.dump_flags(verbose)
        return self

    def reset(self, mol=None):
        self.with_solvent.reset(mol)
        return super().reset(mol)

    def kernel(self, *args, **kwargs):
        with_solvent = self.with_solvent
        # The underlying ._scf object is decorated with solvent effects.
        # The resultant Fock matrix and orbital energies both include the
        # effects from solvent. It means that solvent effects for post-HF
        # methods are automatically counted if solvent is enabled at scf
        # level.
        if with_solvent.frozen:
            return super().kernel(*args, **kwargs)

        log = logger.new_logger(self)
        log.info('\n** Self-consistently update the solvent effects for %s **',
                 self.__class__.__name__)
        ##TODO: Suppress a few output messages
        #log1 = copy.copy(log)
        #log1.note, log1.info = log1.info, log1.debug

        basic_scanner = self._basic_scanner
        e_last = 0
        #diis = lib.diis.DIIS()
        for cycle in range(self.with_solvent.max_cycle):
            log.info('\n** Solvent self-consistent cycle %d:', cycle)
            # Solvent effects are applied when accessing the
            # underlying ._scf objects. The flag frozen=True ensures that
            # the generated potential with_solvent.v is passed to the
            # the post-HF object, without being updated in the implicit
            # call to the _scf iterations.
            with lib.temporary_env(with_solvent, frozen=True):
                e_tot = basic_scanner(self.mol)
                dm = basic_scanner.make_rdm1(ao_repr=True)
                #dm = diis.update(dm)

            # To generate the solvent potential for ._scf object. Since
            # frozen is set when calling basic_scanner, the solvent
            # effects are frozen during the scf iterations.
            with_solvent.e, with_solvent.v = with_solvent.kernel(dm)

            de = e_tot - e_last
            log.info('Solvent cycle %d  E_tot = %.15g  dE = %g',
                     cycle, e_tot, de)

            if abs(e_tot-e_last).max() < with_solvent.conv_tol:
                break
            e_last = e_tot

        # An extra cycle to compute the total energy
        log.info('\n** Extra cycle for solvent effects')
        with lib.temporary_env(with_solvent, frozen=True):
            #Update everything except the _scf object and _keys
            basic_scanner(self.mol)
            self.__dict__.update(basic_scanner.__dict__)
            self._scf.__dict__.update(basic_scanner._scf.__dict__)
        self._finalize()
        return self.e_corr, None

    def nuc_grad_method(self):
        logger.warn(self, '''
Approximate gradients are evaluated here. A small error may be expected in the
gradients which corresponds to the contribution of
DM * V_solvent[d/dX DM] + V_solvent[DM] * d/dX DM
''')
        from pyscf.solvent.grad.pcm import make_grad_object
        return make_grad_object(self)

    Gradients = nuc_grad_method

    def to_gpu(self):
        obj = self.undo_solvent().to_gpu()
        obj = _for_post_scf(obj, self.with_solvent)
        return lib.to_gpu(self, obj)


def _for_tdscf(method, solvent_obj=None, dm=None, equilibrium_solvation=False):
    '''Add solvent model in TDDFT calculations.

    Kwargs:
        dm : if given, solvent does not respond to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''
    assert hasattr(method._scf, 'with_solvent')
    if method._scf.with_solvent.__class__.__name__ == 'PolEmbed':
        # PolEmbed is currently not compatible with the implicit solvent implementation
        from pyscf.solvent.pol_embed import pe_for_tdscf
        return pe_for_tdscf(method, solvent_obj, dm, equilibrium_solvation)

    if solvent_obj is None:
        if isinstance(method, _Solvation):
            return method

        solvent_obj = method._scf.with_solvent.copy()
        solvent_obj.equilibrium_solvation = equilibrium_solvation
        if not equilibrium_solvation:
            # The vertical excitation is a fast process, applying non-equilibrium
            # solvation with optical dielectric constant eps=1.78
            # TODO: reset() can be skipped. Most intermeidates can be reused.
            solvent_obj.reset()
            solvent_obj.eps = 1.78
            solvent_obj.build()

    if isinstance(method, _Solvation):
        method = method.copy()
        method.with_solvent = solvent_obj
        return method

    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True
        if solvent_obj.equilibrium_solvation:
            raise RuntimeError(
                '"frozen" solvent model conflicts to the assumption of equilibrium solvation.')

    sol_td = TDSCFWithSolvent(method, solvent_obj)
    name = solvent_obj.__class__.__name__ + method.__class__.__name__
    return lib.set_class(sol_td, (TDSCFWithSolvent, method.__class__), name)

class TDSCFWithSolvent(_Solvation):
    '''LR Solvent for TDDFT.

    Note: This class does not support the state-specific excited state solvent.
    '''

    _keys = {'with_solvent'}

    def __init__(self, method, solvent_obj):
        self.__dict__.update(method.__dict__)
        self.with_solvent = solvent_obj

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, TDSCFWithSolvent, name_mixin))
        obj._scf = self._scf.undo_solvent()
        return obj

    @property
    def equilibrium_solvation(self):
        '''Whether to allow the solvent rapidly responds to the changes of
        electronic structure or geometry of solute.
        '''
        return self.with_solvent.equilibrium_solvation
    @equilibrium_solvation.setter
    def equilibrium_solvation(self, val):
        if val and self.with_solvent.frozen:
            raise RuntimeError(
                '"frozen" Solvent model was set in the '
                'ground state SCF calculation. It conflicts to '
                'the assumption of equilibrium solvation.\n'
                'You can set _scf.with_solvent.frozen = False and '
                'rerun the ground state calculation _scf.run().')
        self.with_solvent.equilibrium_solvation = val

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info('Solvent model for TDDFT:')
        self.with_solvent.check_sanity()
        self.with_solvent.dump_flags(verbose)
        return self

    def reset(self, mol=None):
        self.with_solvent.reset(mol)
        return super().reset(mol)

    def gen_response(self, *args, **kwargs):
        # vind computes the response in gas-phase
        vind = self._scf.undo_solvent().gen_response(
            *args, with_nlc=not self.exclude_nlc, **kwargs)

        # The contribution of the solvent to an excited state include the fast
        # and the slow response parts. In the process of fast vertical excitation,
        # only the fast part is able to respond to changes of the solute
        # wavefunction. This process is described by the non-equilibrium
        # solvation. In the excited Hamiltonian, the potential from the slow part is
        # omitted. Changes of the solute electron density would lead to a
        # redistribution of the surface charge (due to the fast part).
        # The redistributed surface charge is computed by solving
        #     K^{-1} R (dm_response)
        # using a different dielectric constant. The optical dielectric constant
        # (eps=1.78, see QChem manual) is a suitable choice for the excited state.
        if not self.with_solvent.equilibrium_solvation:
            # Solvent with optical dielectric constant, for evaluating the
            # response of the fast solvent part
            with_solvent = self.with_solvent
            logger.info(self, 'TDDFT non-equilibrium solvation with eps=%g', with_solvent.eps)
        else:
            # Solvent with zero-frequency dielectric constant. The ground state
            # solvent is utilized to ensure the same eps are used in the
            # gradients of excited state.
            with_solvent = self._scf.with_solvent
            logger.info(self, 'TDDFT equilibrium solvation with eps=%g', with_solvent.eps)

        is_uhf = isinstance(self._scf, scf.uhf.UHF)
        singlet = kwargs.get('singlet', True)
        singlet = singlet or singlet is None
        def vind_with_solvent(dm1):
            v = vind(dm1)
            if is_uhf:
                v += with_solvent._B_dot_x(dm1[0]+dm1[1])
            elif singlet:
                v += with_solvent._B_dot_x(dm1)
            else:
                # The triplet excitation does not change the total electron
                # density, thus does not lead to solvent response.
                pass
            return v
        return vind_with_solvent

    def get_ab(self, mf=None):
        raise NotImplementedError

    def nuc_grad_method(self):
        from pyscf.solvent.pcm import PCM
        from pyscf.solvent.grad.ddcosmo_tdscf_grad import make_grad_object
        if isinstance(self.with_solvent, PCM):
            raise NotImplementedError('PCM-TDDFT Gradients')
        return make_grad_object(self)

    Gradients = nuc_grad_method

    def to_gpu(self):
        obj = self.undo_solvent().to_gpu()
        obj = _for_tdscf(obj, self.with_solvent)
        return lib.to_gpu(self, obj)
