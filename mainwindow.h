#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    boost::filesystem::path ImageFolder;
    boost::filesystem::path OutFolder;

    std::string FileName;
    std::string FileNameTxt;
    cv::Mat ImIn;
    cv::Mat ImGray;
    cv::Mat ImGradient;
    cv::Mat Mask;
    cv::Mat MaskHair;
    cv::Mat ImOut;

    cv::Mat LesionMask;

    std::vector<cv::Mat> TileImVector;
    std::vector<cv::Mat> TileMaskVector;
    std::vector<cv::Point> TilePositionVector;



    double minIm;
    double maxIm;

    double displayScale;

    void OpenImageFolder();
    void ReadImage();
    void ShowsScaledImage(cv::Mat Im, std::string ImWindowName);
    void ShowImages();
    void ProcessImages();
    void ProcessTile();
    void ProcessTile2();


private slots:
    void on_pushButtonOpenImageFolder_clicked();

    void on_lineEditRegexImageFile_returnPressed();

    void on_listWidgetImageFiles_currentTextChanged(const QString &currentText);

    void on_checkBoxShowInput_toggled(bool checked);

    void on_checkBoxShowTiffInfo_toggled(bool checked);

    void on_checkBoxShowMatInfo_toggled(bool checked);

    void on_checkBoxAutocleanOut_stateChanged(int arg1);

    void on_checkBoxLoadAnydepth_toggled(bool checked);

    void on_checkBoxShowOutput_toggled(bool checked);

    void on_spinBoxScalePower_valueChanged(int arg1);

    void on_spinBoxScaleBase_valueChanged(int arg1);

    void on_checkBoxImRotate_toggled(bool checked);

    void on_pushButtonOpenOutFolder_clicked();

    void on_doubleSpinBoxFixMinDisp_valueChanged(double arg1);

    void on_doubleSpinBoxFixMaxDisp_valueChanged(double arg1);

    void on_checkBoxShowGray_toggled(bool checked);

    void on_checkBoxShowGradient_toggled(bool checked);

    void on_doubleSpinBoxFixMinDispGrad_valueChanged(double arg1);

    void on_doubleSpinBoxFixMaxDispGrad_valueChanged(double arg1);

    void on_spinBoxGradientSchape_valueChanged(int arg1);

    void on_spinBoxGradThreshold_valueChanged(int arg1);

    void on_comboBoxGradient_currentIndexChanged(int index);

    void on_checkBoxShowMask_toggled(bool checked);

    void on_comboBoxOutputMode_currentIndexChanged(int index);

    void on_checkBoxShowPC_toggled(bool checked);

    void on_spinBoxTileToProcess_valueChanged(int arg1);

    void on_spinBoxHistScaleHeight_valueChanged(int arg1);

    void on_spinBoxHistScaleCoef_valueChanged(int arg1);

    void on_spinBoxHistBarWidth_valueChanged(int arg1);


    void on_checkBoxShowHist_toggled(bool checked);

    void on_checkBoxSaveOutput_toggled(bool checked);

    void on_comboBoxDisplayRange_currentIndexChanged(int index);

    void on_checkBoxProcessGradient_toggled(bool checked);

    void on_checkBoxMaskBackGround_toggled(bool checked);

    void on_checkBoxProcessTile_toggled(bool checked);

    void on_spinBoxTileSize_valueChanged(int arg1);

    void on_checkBoxShowTile_toggled(bool checked);

    void on_doubleSpinBoxTileScale_valueChanged(double arg1);

    void on_checkBoxShowTileOnImage_toggled(bool checked);

    void on_checkBoxShowLesionMask_toggled(bool checked);

    void on_spinBoxTileSizeX_valueChanged(int arg1);



    void on_pushButtonSaveTiles_clicked();

    void on_pushButtonAnalyse_clicked();

    void on_spinBoxTileX_valueChanged(int arg1);

    void on_spinBoxTileY_valueChanged(int arg1);

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
