import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { Layout, Menu, Button } from 'antd';
import { SettingOutlined } from '@ant-design/icons';
import ProblemWorkbench from './pages/ProblemWorkbench';
import RunMonitoring from './pages/RunMonitoring';
import Dashboard from './pages/Dashboard';
import ProblemDetails from './pages/ProblemDetails';
import LLMConfigModal from './components/LLMConfigModal';
import DAGSetup from './pages/DAGSetup';
import EnvironmentSetup from './pages/EnvironmentSetup';
import Planner from './pages/Planner';
import RunAnalysis from './pages/RunAnalysis';

const { Header, Content, Footer } = Layout;

// 1. 创建并导出 Context
export const AppContext = React.createContext(null);

function App() {
    const [isModalOpen, setIsModalOpen] = useState(false);

    return (
        <AppContext.Provider value={{ setIsModalOpen }}>
            <Router>
                <Layout style={{ minHeight: '100vh' }}>
                    <Header style={{ display: 'flex', alignItems: 'center' }}>
                        <div style={{ color: 'white', fontSize: '20px' }}>
                            ISEC LABORATORY
                        </div>
                        <Menu
                            theme="dark"
                            mode="horizontal"
                            defaultSelectedKeys={["1"]}
                            style={{ flex: 1, minWidth: 0, marginLeft: '24px', marginRight: '24px' }}
                            items={[
                                {
                                    key: '1',
                                    label: <Link to="/">Dashboard</Link>,
                                },
                                {
                                    key: '3',
                                    label: <Link to="/workflow">Workflow Setup</Link>,
                                },
                            ]}
                        />
                        <Button
                            type="primary"
                            icon={<SettingOutlined />}
                            onClick={() => setIsModalOpen(true)}
                        >
                            LLM Settings
                        </Button>
                    </Header>
                    <Content style={{ padding: '20px 50px' }}>
                        <div style={{ background: '#fff', padding: 24, minHeight: 280 }}>
                            <Routes>
                                <Route path="/" element={<Dashboard />} />
                                <Route path="/workflow" element={<DAGSetup />} />
                                <Route path="/environment" element={<EnvironmentSetup />} />
                                <Route path="/problem/:problemId" element={<ProblemDetails />} />
                                <Route path="/run/:runId" element={<RunMonitoring />} />
                                <Route path="/planner/:runId" element={<Planner />} />
                                <Route path="/analysis/:runId" element={<RunAnalysis />} />
                            </Routes>
                        </div>
                    </Content>
                    <Footer style={{ textAlign: 'center' }}>
                        ISEC LABORATORY
                    </Footer>
                </Layout>
                <LLMConfigModal
                    open={isModalOpen}
                    onOk={() => setIsModalOpen(false)}
                    onCancel={() => setIsModalOpen(false)}
                />
            </Router>
        </AppContext.Provider>
    );
}

export default App;
